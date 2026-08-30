import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms

import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os

from PIL import Image
from IPython.display import display, clear_output

from transformers import CLIPTokenizer

from module.vae import VAE_Encoder
from module.clip import CLIP
from module.diffusion import DiffusionModel
from module.pipeline import get_time_embedding, generate
from module.model_loader import get_model
from module.samplers.sampler import Sampler

from trainer.dataset import ImageTextDataset

import warnings;warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

class DiffusionTrainer:
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "../checkpoint/diffusion",
        vae_ckp: str = "../checkpoint/vae/vae_epoch_90.pt",
        clip_ckp: str = "../model/base-v1-5-pruned-emaonly.safetensors",
        diffusion_ckp: str = "../model/base-v1-5-pruned-emaonly.safetensors",
        tokenizer_dir: str = "../tokenizer",
        image_size: int = 128,
        batch_size: int = 8,
        num_workers: int = 4,
        learning_rate: float = 1e-5,
        num_epochs: int = 100,
        num_train_steps: int = 1000,
        prompt_dropout: float = 0.1,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        save_every: int = 5,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):

        self.output_dir = output_dir
        self.image_size = image_size
        self.latent_height = image_size // 8
        self.latent_width = image_size // 8
        self.num_epochs = num_epochs
        self.num_train_steps = num_train_steps
        self.prompt_dropout = prompt_dropout
        self.save_every = save_every
        self.device = device
        self.dtype = dtype
        os.makedirs(output_dir, exist_ok=True)

        # ========================= 1. 初始化模型 =========================
        print("🔄 Initializing CLIPTokenizer...")
        self.tokenizer = CLIPTokenizer(f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt")
        print("✅ Finished!")
        print("=" * 80)

        self.encoder = get_model(name='encoder', ckp_path=vae_ckp, device=device, dtype=dtype)
        self.decoder = get_model(name='decoder', ckp_path=vae_ckp, device=device, dtype=dtype)
        self.clip = get_model(name='clip', ckp_path=clip_ckp, device=device, dtype=dtype)
        self.diffusion = get_model(name='diffusion', ckp_path=diffusion_ckp, device=device, dtype=dtype)
        print("=" * 80)
        
        print("🔄 Frozen encoder, decoder and clip...")
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False
        print("✅ Finished!")
        print("=" * 80)

        # ========================= 2. 准备数据集 =========================
        print("📂 Loading dataset...")
        self.dataset = ImageTextDataset(
            data_dir=data_dir,
            image_size=image_size,
            tokenizer=self.tokenizer,
            max_length=77,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device == "cuda" else False,
            drop_last=True,
        )

        # Show the first three pictures
        fig, axs = plt.subplots(1, 3, figsize=(8, 2))
        for idx, ax in enumerate(axs):
            img = self.dataset[idx]["image"]
            img = (img.permute(1, 2, 0) + 1) / 2
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        for idx in range(3):
            print(f"Caption {idx + 1}: {self.dataset[idx]['caption']}")
            print(f"Tokenized {idx + 1}: {self.dataset[idx]['input_ids']}")
        print("=" * 80)

        # ========================= 3. 加载优化器、调度器、采样器等 =========================
        print("🔄 Initializing optimizer, scheduler, scaler, sampler, etc...")
        self.optimizer = AdamW(
            self.diffusion.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-7
        )
        self.scaler = GradScaler("cuda") if device == "cuda" else None
        self.noise_generator = torch.Generator(device=device)
        self.sampler = Sampler(
            generator=self.noise_generator,
            num_training_steps=num_train_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        ).to(device)
        self.epoch = 0
        self.ckp_epochs = 0
        print("✅ Finished!")
        print("=" * 80)
        
        # ========================= 4. 打印配置信息 =========================
        print("\n📊 Configuration:")
        print(f"   DATASET DIRECTORY:    {data_dir}")
        print(f"   DATASET SIZE:         {len(self.dataset)}")
        print(f"   CHECKPOINT DIRECTORY: {output_dir}")
        print(f"   VAE CHECKPOINT:       {vae_ckp}")
        print(f"   CLIP CHECKPOINT:      {clip_ckp}")
        print(f"   DIFFUSION CHECKPOINT: {diffusion_ckp}")
        print(f"   IMAGE SIZE:           {image_size}x{image_size}")
        print(f"   LATENT SIZE:          {self.latent_height}x{self.latent_width}")
        print(f"   BATCH SIZE:           {batch_size}")
        print(f"   NUM WORKERS:          {num_workers}")
        print(f"   LEARNING RATE:        {learning_rate}")
        print(f"   TOTAL EPOCHS:         {num_epochs}")
        print(f"   TRAIN STEPS:          {num_train_steps}")
        print(f"   PROMPT DROPOUT:       {prompt_dropout}")
        print(f"   SAVE INTERVAL:        {save_every}")
        print(f"   DEVICE:               {device}")
        print(f"   DTYPE:                {dtype}")
        print("="*80)
        print("✅ **All initialization is ready, start training using trainer.train()**")

    def train_epoch(self):
        epoch_loss = 0.0
        progress_bar = tqdm(
            self.dataloader,
            desc=f"Epoch {self.epoch + self.ckp_epochs}/{self.num_epochs + self.ckp_epochs}",
        )

        for batch in progress_bar:
            # 当前批次训练数据
            # images shape: (batch_size, 3, height, width)
            images = batch["image"].to(self.device)
            
            # input_ids shape: (batch_size, 77)
            input_ids = batch["input_ids"].to(self.device)
            batch_size = images.shape[0]

            with torch.no_grad():
                ##### ========================= 1. VAE编码，生成 latents 特征图 =======================
                # vae_noise shape:       (batch_size, 4, height / 8, width / 8)
                # latents shape:         (batch_size, 4, height / 8, width / 8)
                vae_noise = torch.randn(batch_size, 4, self.latent_height, self.latent_width, device=self.device, dtype=self.dtype)
                latents = self.encoder(images, vae_noise)

                ##### =============== 2. 随机采样时间步，根据 DDPM 前向公式对 latent 加噪 ===============
                # timesteps shape:       (batch_size, )
                # latents shape:         (batch_size, 4, height / 8, width / 8)
                timesteps = torch.randint(0, self.num_train_steps, (batch_size,), device=self.device, dtype=torch.long)
                noise = torch.randn(batch_size, 4, self.latent_height, self.latent_width, device=self.device, dtype=self.dtype)
                noisy_latents = self.sampler.add_noise(latents, timesteps, noise)
    
                ##### =========================== 3. CLIP 编码提示词和时间步 ===========================
                # 随机丢弃提示词
                if random.random() < self.prompt_dropout:
                    # 用空 prompt 的 token 替换当前 batch 的 input_ids
                    empty_tokens = self.tokenizer(
                        [""] * batch_size,
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = empty_tokens["input_ids"].to(self.device)
                # context shape:         (batch_size, 77, 768)
                # time_embeddings shape: (batch_size, 1, 320)
                context = self.clip(input_ids)
                time_embeddings = torch.stack([get_time_embedding(t.item(), device=self.device, dtype=self.dtype) for t in timesteps]).squeeze(1).to(self.device)

            ##### ========================= 4. 前向传播，DiffusionModel 预测噪声 =========================
            self.optimizer.zero_grad()
            with autocast("cuda"):
                predicted_noise = self.diffusion(noisy_latents, context, time_embeddings)
                loss = F.mse_loss(predicted_noise, noise)

            ##### ================================= 5. 反向传播及更新参数 =================================
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            ##### ================================ 6. 累计损失并更新进度条 ================================
            loss_item = loss.item()
            epoch_loss += loss_item
            progress_bar.set_postfix({"loss": f"{loss_item:.4f}"})

        return epoch_loss / len(self.dataloader)

    def train(self):
        print(f"🚀 Starting DiffusionModel training for {self.num_epochs} epochs...")

        for _ in range(self.num_epochs):
            self.epoch += 1

            # 训练模式
            self.diffusion.train()
            # 训练一个 epoch
            avg_loss = self.train_epoch()
            # 学习率调度
            self.scheduler.step()

            print(f"\n📊 Epoch {self.ckp_epochs + self.epoch} summary：")
            print(f"   Average Loss: {avg_loss:.6f}")
            print(f"   LR:           {self.scheduler.get_last_lr()[0]:.2e}")

            # 随机挑选一条样本进行可视化：编码 latent -> 加噪 -> 预测噪声 -> 去噪 -> 解码
            self.visualize_prediction()

            # 保存检查点
            if self.epoch % self.save_every == 0 or self.epoch == self.num_epochs:
                self.save_checkpoint()

        print("✅ Training finished!")

    def visualize_prediction(self):
        self.diffusion.eval()

        # 从图像和对应的提示词、以及时间步中随机抽样
        idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]
        
        # 输入图像
        input_image = sample["image"].to(self.device)
        input_image = transforms.ToPILImage()((input_image + 1) / 2)
        
        # 提示词
        prompt = sample["caption"]
        
        # 从 0~num_train_steps 随机选取一步并转化为比例，即对输入图像加噪的程度
        strength = torch.randint(0, self.num_train_steps, (1,), device=self.device, dtype=torch.long)[0] / 1000
        
        # 模型
        models = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "clip": self.clip,
            "diffusion": self.diffusion.eval(),
        }
        
        # 推理循环
        generator = generate(
            prompt=prompt,
            uncond_prompt=None,
            img_width=self.image_size,
            img_height=self.image_size,
            input_image=input_image,
            strength=strength,
            do_cfg=False,
            n_inference_steps=64,
            num_training_steps=self.num_train_steps,
            models=models,
            seed=random.randint(0, int(1e8)),
            device=self.device,
            dtype=self.dtype,
            tokenizer=self.tokenizer,
        )
        
        # 获取降噪完成后的图像
        for output_image in generator:
            pass
            
        # 绘制
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(input_image)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(output_image[0])
        axs[1].set_title("Denoised Image")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()

        self.diffusion.train()

    def save_checkpoint(self, path: str = None, name: str = None):
        checkpoint = {
            "epoch": self.epoch + self.ckp_epochs,
            "diffusion": self.diffusion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        save_path = os.path.join(
            path if path else self.output_dir,
            name if name else f"diffusion_epoch_{self.ckp_epochs + self.epoch}.pt",
        )
        torch.save(checkpoint, save_path)
        
        print(f"💾 Checkpoint saved: {save_path}")

    def load_checkpoint(self, path: str):
        print("🔄 Loading checkpoint...")
        checkpoint = torch.load(path, map_location=self.device)

        self.ckp_epochs = checkpoint["epoch"]
        self.diffusion.load_state_dict(checkpoint["diffusion"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        print(f"📂 Checkpoint loaded: {path} (Epoch {self.ckp_epochs})")