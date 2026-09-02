import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
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
from module.model_loader import get_encoder, get_decoder, get_clip, get_diffusion
from module.samplers.sampler import Sampler
from trainer.dataset import ImageTextDataset

import warnings;warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

class DiffusionTrainer:
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "../checkpoint/diffusion",
        vae_ckp: str = "../model/v1-5-pruned-emaonly.safetensors",
        clip_ckp: str = "../model/v1-5-pruned-emaonly.safetensors",
        diffusion_ckp: str = "../model/v1-5-pruned-emaonly.safetensors",
        tokenizer_dir: str = "../tokenizer",
        image_size: int = 512,                   # 训练图像大小
        batch_size: int = 8,                     # 批次大小
        accumulation_steps: int = 4,             # 梯度积累间隔
        num_workers: int = 16,                   # DataLoader 子进程数   
        learning_rate: float = 1e-5,             # 学习率
        eta_min: float = 1e-7,                   # 调度器的最低学习率
        num_epochs: int = 30,                    # 训练总轮次
        save_every: int = 3,                     # 检查点保存间隔
        val_split: float = 0.1,                  # 验证集分割比例
        split_seed: int = 42,                    # 确保验证集分割可复现
        use_ema: bool = True,                    # 使用 EMA 影子模型
        ema_decay: float = 0.9999,               # EMA 衰减系数
        prompt_dropout: float = 0.1,             # CFG 训练提示词随机丢弃比率
        device: str = "cuda",                    # 训练设备
        dtype: torch.dtype = torch.float32,      # 默认 float32，自动启用混合精度

        ##### 以下参数为预训练参数，保持官方默认    
        num_train_steps: int = 1000,             # 降噪总轮次
        beta_start: float = 0.00085,             # 采样策略参数
        beta_end: float = 0.0120,                # 采样策略参数
        scaling_factor: float = 0.18215,         # VAE Rescaling 系数
    ):

        self.output_dir = output_dir
        self.image_size = image_size
        self.latent_height = image_size // 8
        self.latent_width = image_size // 8
        self.accumulation_steps = max(1, accumulation_steps)
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_updates = 0
        self.prompt_dropout = prompt_dropout
        self.device = device
        self.dtype = dtype
        self.num_train_steps = num_train_steps
        self.scaling_factor = scaling_factor

        os.makedirs(output_dir, exist_ok=True)

        # ========================= 1. 模型初始化 =========================
        # 初始化分词器
        print("🔄 Initializing CLIPTokenizer...")
        self.tokenizer = CLIPTokenizer(f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt")
        print("> Finished!")
        
        print("=" * 80)

        # 初始化 VAE、CLIP、Diffusion
        self.encoder = get_encoder(ckp_path=vae_ckp, device=device, dtype=dtype)
        self.decoder = get_decoder(ckp_path=vae_ckp, device=device, dtype=dtype)
        self.clip = get_clip(ckp_path=clip_ckp, device=device, dtype=dtype)
        self.diffusion = get_diffusion(ckp_path=diffusion_ckp, device=device, dtype=dtype)

        print("=" * 80)
        
        # 初始化 EMA 影子模型
        if self.use_ema:
            print("🔄 Initializing EMA shadow model for Diffusion...")
            self.ema_diffusion = DiffusionModel().to(device).to(dtype)
            self.ema_diffusion.load_state_dict(self.diffusion.state_dict())
            self.ema_diffusion.eval()
            for param in self.ema_diffusion.parameters():
                param.requires_grad = False
            print("> Finished")
        else:
            self.ema_diffusion = None

        print("=" * 80)

        # 冻结 VAE 和 CLIP，仅训练 Diffusion 模型
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
        print("> Finished!")
        
        print("=" * 80)

        # ========================= 2. 准备数据集 =========================
        print("📂 Loading dataset...")
        self.dataset = ImageTextDataset(
            data_dir=data_dir,
            image_size=image_size,
            tokenizer=self.tokenizer,
            max_length=77,
        )

        # 按 val_split 比例拆分为训练集和验证集，使用 split_seed 保证结果可复现
        dataset_size = len(self.dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        
        print(f"🔄 Spliting into train dataset({train_size} images) and val dataset({val_size} images)...")
        generator = torch.Generator().manual_seed(split_seed)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=generator,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device == "cuda" else False,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device == "cuda" else False,
            drop_last=False,
        )

        # 从训练集展示前三张图像
        fig, axs = plt.subplots(1, 3, figsize=(8, 2))
        for idx, ax in enumerate(axs):
            img = self.train_dataset[idx]["image"]
            img = (img.permute(1, 2, 0) + 1) / 2
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        for idx in range(3):
            print(f"Caption {idx + 1}: {self.train_dataset[idx]['caption']}")
            print(f"Tokenized {idx + 1}: {self.train_dataset[idx]['input_ids']}")
        
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
            self.optimizer, 
            T_max=num_epochs, 
            eta_min=eta_min
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
        print("> Finished!")
        
        print("=" * 80)
        
        # ========================= 4. 打印配置信息 =========================
        print("\n📊 Configuration:")
        print(f"   DATASET DIRECTORY:    {data_dir}")
        print(f"   DATASET SIZE:         {len(self.dataset)}")
        print(f"   TRAIN SET SIZE:       {len(self.train_dataset)} ({1 - val_split:.0%})")
        print(f"   VAL SET SIZE:         {len(self.val_dataset)} ({val_split:.0%})")
        print(f"   CHECKPOINT DIRECTORY: {output_dir}")
        print(f"   VAE CHECKPOINT:       {vae_ckp}")
        print(f"   CLIP CHECKPOINT:      {clip_ckp}")
        print(f"   DIFFUSION CHECKPOINT: {diffusion_ckp}")
        print(f"   IMAGE SIZE:           {image_size}x{image_size}")
        print(f"   LATENT SIZE:          {self.latent_height}x{self.latent_width}")
        print(f"   BATCH SIZE:           {batch_size}")
        print(f"   ACCUMULATION STEPS:   {accumulation_steps}")
        print(f"   NUM WORKERS:          {num_workers}")
        print(f"   LEARNING RATE:        {learning_rate}")
        print(f"   MIN LEARNING RATE:    {eta_min}")
        print(f"   TOTAL EPOCHS:         {num_epochs}")
        print(f"   VAL SPLIT:            {val_split}")
        print(f"   SPLIT SEED:           {split_seed}")
        print(f"   USE EMA:              {use_ema}")
        print(f"   EMA DECAY:            {ema_decay}")
        print(f"   SAVE INTERVAL:        {save_every}")
        print(f"   PROMPT DROPOUT:       {prompt_dropout}")
        print(f"   TRAIN STEPS:          {num_train_steps}")
        print(f"   BETA START:           {beta_start}")
        print(f"   BETA END:             {beta_end}")
        print(f"   SCALING FACTOR:       {scaling_factor}")
        print(f"   DEVICE:               {device}")
        print(f"   DTYPE:                {dtype}")
        
        print("="*80)
        
        print("✅ **All initialization is ready, start training using trainer.train()**")

    def train_epoch(self):
        """训练一个 Epoch，返回所有批次的平均损失。"""
        
        epoch_loss = 0.0 # 当前轮次损失
        step_count = 0   # 梯度积累判断

        # 进度条
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + self.ckp_epochs}/{self.num_epochs + self.ckp_epochs}")

        # 每个 epoch 开始时清空梯度，避免上一个 epoch 末残留的梯度影响
        self.optimizer.zero_grad()

        # 训练循环
        for batch_idx, batch in enumerate(progress_bar):
            # 获取当前批次训练数据
            # images shape: (batch_size, 3, height, width)
            # input_ids shape: (batch_size, 77)
            images = batch["image"].to(device=self.device, dtype=self.dtype)
            input_ids = batch["input_ids"].to(device=self.device)
            batch_size = images.shape[0]

            with torch.no_grad():
                ##### 1. VAE编码，生成 latents 特征图
                # vae_noise shape:       (batch_size, 4, height // 8, width // 8)
                # latents shape:         (batch_size, 4, height // 8, width // 8)
                # 传入全零张量，使解码器直接输出均值
                vae_noise = torch.zeros(batch_size, 4, self.latent_height, self.latent_width, device=self.device, dtype=self.dtype)
                latents = self.encoder(images, vae_noise, self.scaling_factor)

                ##### 2. 随机采样时间步，根据 DDPM 前向公式对 latent 加噪
                # timesteps shape:       (batch_size, )
                # latents shape:         (batch_size, 4, height // 8, width // 8)
                timesteps = torch.randint(0, self.num_train_steps, (batch_size,), device=self.device, dtype=torch.long)
                noise = torch.randn(batch_size, 4, self.latent_height, self.latent_width, device=self.device, dtype=self.dtype)
                noisy_latents = self.sampler.add_noise(latents, timesteps, noise)

                ##### 3. CLIP 编码提示词和时间步
                # 每个样本有 prompt_dropout 的概率丢弃
                drop_mask = torch.rand(batch_size) < self.prompt_dropout
                if drop_mask.any():
                    # 用空 prompt 的 token 替换当前 batch 的 input_ids
                    empty_tokens = self.tokenizer([""] * int(drop_mask.sum()), padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                    input_ids[drop_mask] = empty_tokens["input_ids"].to(self.device)
                # context shape:         (batch_size, 77, 768)
                # time_embeddings shape: (batch_size, 1, 320)
                context = self.clip(input_ids)
                time_embeddings = torch.stack([get_time_embedding(t.item(), device=self.device, dtype=self.dtype) for t in timesteps]).squeeze(1).to(self.device)

            ##### 4. 前向传播，Diffusion 模型预测噪声
            with autocast("cuda", dtype=torch.bfloat16):
                predicted_noise = self.diffusion(noisy_latents, context, time_embeddings)
                # 将损失按积累步数缩放，使得多步积累的梯度等效于一个大批次
                loss = F.mse_loss(predicted_noise, noise) / self.accumulation_steps

            ##### 5. 反向传播，不立即更新参数
            self.scaler.scale(loss).backward()

            ##### 6. 积累步数，当完成一个积累周期，或到达 epoch 末尾时，执行一次参数更新
            is_last_batch = (batch_idx + 1) == len(self.train_dataloader)
            if (batch_idx + 1) % self.accumulation_steps == 0 or is_last_batch:
                # 更新 diffusion 模型
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # 更新 EMA 影子模型
                self.update_ema()

                step_count += 1

            ##### 8. 累计损失并更新进度条
            # 进度条显示的是反缩放后的单步真实损失
            loss_item = loss.item() * self.accumulation_steps
            epoch_loss += loss_item
            progress_bar.set_postfix({"loss": f"{loss_item:.4f}"})

        # 返回所有 batch 的平均损失
        return epoch_loss / len(self.train_dataloader)

    @torch.no_grad()
    def update_ema(self):
        """更新 EMA 影子模型参数。"""

        # 未启用 EMA 则直接返回
        if not self.use_ema or self.ema_diffusion is None:
            return

        # warmup 预热: 前期 decay 较小，让 EMA 更快跟上当前模型，随着更新次数增加，decay 逐渐接近 ema_decay
        self.ema_updates += 1
        decay = min(self.ema_decay, (1 + self.ema_updates) / (10 + self.ema_updates))

        # 加权平均更新 EMA
        for ema_param, param in zip(self.ema_diffusion.parameters(), self.diffusion.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @torch.no_grad()
    def validate(self, use_ema: bool = None):
        """在验证集上评估 Diffusion 模型的噪声预测损失。"""
        
        if use_ema is None:
            use_ema = self.use_ema

        # 选择评估模型
        if use_ema and self.ema_diffusion is not None:
            model = self.ema_diffusion
            model.eval()
        else:
            model = self.diffusion
            self.diffusion.eval()

        val_loss = 0.0
        progress_bar = tqdm(self.val_dataloader, desc=f"Validation {'(EMA)' if use_ema else '(Online)'} {self.epoch + self.ckp_epochs}/{self.num_epochs + self.ckp_epochs}")

        for batch in progress_bar:
            images = batch["image"].to(device=self.device, dtype=self.dtype)
            input_ids = batch["input_ids"].to(device=self.device)
            batch_size = images.shape[0]

            ##### 1. VAE 编码
            vae_noise = torch.zeros(batch_size, 4, self.latent_height, self.latent_width, device=self.device, dtype=self.dtype)
            latents = self.encoder(images, vae_noise, self.scaling_factor)

            ##### 2. 前向加噪
            timesteps = torch.randint(0, self.num_train_steps, (batch_size,), device=self.device, dtype=torch.long)
            noise = torch.randn(batch_size, 4, self.latent_height, self.latent_width, device=self.device, dtype=self.dtype)
            noisy_latents = self.sampler.add_noise(latents, timesteps, noise)

            ##### 3. CLIP 编码提示词
            context = self.clip(input_ids)
            time_embeddings = torch.stack([get_time_embedding(t.item(), device=self.device, dtype=self.dtype) for t in timesteps]).squeeze(1).to(self.device)

            ##### 4. 预测噪声并计算损失
            predicted_noise = model(noisy_latents, context, time_embeddings)
            loss = F.mse_loss(predicted_noise, noise)

            ##### 5. 积累损失并更新进度条
            val_loss += loss.item()
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        # 恢复训练状态
        if not (use_ema and self.ema_diffusion is not None):
            self.diffusion.train()
        
        return val_loss / len(self.val_dataloader)

    def train(self):
        print(f"🚀 Starting DiffusionModel training for {self.num_epochs} epochs...")

        for _ in range(self.num_epochs):
            self.epoch += 1

            # 恢复训练状态
            self.diffusion.train()
            
            # 训练一个 Epoch
            avg_loss = self.train_epoch()
            
            # 更新学习率调度
            self.scheduler.step()

            print(f"\n📊 Epoch {self.ckp_epochs + self.epoch} summary：")
            print(f"   Average Loss: {avg_loss:.6f}")
            print(f"   LR:           {self.scheduler.get_last_lr()[0]:.2e}")

            # 可视化
            self.visualize_prediction()

            # 保存检查点
            if self.epoch % self.save_every == 0 or self.epoch == self.num_epochs:
                self.save_checkpoint()

        print("✅ Training finished!")

    def visualize_prediction(self):
        """随机挑选一条样本进行可视化。"""
        
        self.diffusion.eval()

        # 从验证集中随机抽样，用于观察模型在未见过数据上的去噪效果
        idx = random.randint(0, len(self.val_dataset) - 1)
        sample = self.val_dataset[idx]
        
        # 输入图像
        input_image = sample["image"].to(self.device)
        input_image = transforms.ToPILImage()((input_image + 1) / 2)
        
        # 提示词
        prompt = sample["caption"]
        
        # 从 (0.6~1)*num_train_steps 中随机选取一步并转化为比例，即对输入图像加噪的程度
        strength = torch.randint(int(0.6 * self.num_train_steps), self.num_train_steps, (1,), device=self.device, dtype=torch.long)[0] / self.num_train_steps
        
        # 若启用 EMA，可视化时使用 EMA 模型
        eval_model = self.ema_diffusion if (self.use_ema and self.ema_diffusion is not None) else self.diffusion
        models = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "clip": self.clip,
            "diffusion": eval_model.eval(),
        }
        
        # 推理循环
        generator = generate(
            prompt=prompt,
            uncond_prompt='',
            img_width=self.image_size,
            img_height=self.image_size,
            input_image=input_image,
            strength=strength,
            do_cfg=True,
            cfg_scale=8.5,
            n_inference_steps=64,
            num_training_steps=self.num_train_steps,
            models=models,
            seed=random.randint(0, int(1e8)),
            device=self.device,
            dtype=self.dtype,
            tokenizer=self.tokenizer,
            scaling_factor=self.scaling_factor,
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
        axs[1].set_title(f"(timestep:{int(strength * self.num_train_steps)})\nDenoised Image")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()

        self.diffusion.train()

    def save_checkpoint(self, path: str = None, name: str = None):
        # 保存前先在验证集上评估一次，并输出验证损失
        if self.use_ema and self.ema_diffusion is not None:
            online_val_loss = self.validate(use_ema=False)
            ema_val_loss = self.validate(use_ema=True)
            print(f"📊 Online Validation Loss: {online_val_loss:.6f}")
            print(f"📊 EMA Validation Loss:    {ema_val_loss:.6f}")
            val_loss = ema_val_loss
        else:
            val_loss = self.validate(use_ema=False)
            print(f"📊 Validation Loss: {val_loss:.6f}")

        checkpoint = {
            "epoch": self.epoch + self.ckp_epochs,
            "diffusion": self.diffusion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "val_loss": val_loss,
            "use_ema": self.use_ema,
            "ema_updates": self.ema_updates,
            "ema_diffusion": self.ema_diffusion.state_dict() if self.use_ema and self.ema_diffusion is not None else None,
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

        # 加载 EMA 状态
        if self.use_ema and self.ema_diffusion is not None and checkpoint.get("ema_diffusion") is not None:
            self.ema_diffusion.load_state_dict(checkpoint["ema_diffusion"])
            self.ema_updates = checkpoint.get("ema_updates", 0)
            print(f"> EMA model loaded (ema_updates={self.ema_updates})")

        print(f"📂 Checkpoint loaded: {path} (Epoch {self.ckp_epochs})")