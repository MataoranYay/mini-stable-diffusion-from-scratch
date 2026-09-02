import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import lpips
import os

from module.vae import VAE_Encoder, VAE_Decoder
from module.model_loader import get_encoder, get_decoder
from trainer.dataset import ImageTextDataset

import warnings;warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

class VAELoss(nn.Module):
    """Loss = L1_recon + MSE_recon + λ_lpips * LPIPS + λ_kl * KL"""
    
    def __init__(
        self,
        lpips_weight: float = 0.1,
        kl_weight: float = 1e-6,  # SD VAE uses tiny KL weight — reconstruction quality prioritized, latent not forced to 𝒩(0, I).
        device: str = "cuda",
    ):
        super().__init__()
        
        self.lpips_weight = lpips_weight
        self.kl_weight = kl_weight
        self.lpips = lpips.LPIPS(net='vgg').to(device)

    def forward(self, recon_x, x, mean, log_var):
        ##### 1. 重建损失
        l1_loss = F.l1_loss(recon_x, x)
        mse_loss = F.mse_loss(recon_x, x)
        recon_loss = l1_loss + mse_loss

        ##### 2. 感知损失
        lpips_loss = self.lpips(recon_x, x).mean() if self.lpips_weight > 0 else torch.tensor(0.0, device=device)

        ##### 3. KL 散度: D_KL(q(z|x) || N(0,1))
        # For Gaussian: KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kl_loss = kl_loss / x.shape[0]  # Average over batch

        ##### 4. 总损失
        total_loss = recon_loss + self.lpips_weight * lpips_loss + self.kl_weight * kl_loss

        return {
            "total": total_loss,
            "recon": recon_loss,
            "lpips": lpips_loss if isinstance(lpips_loss, torch.Tensor) else torch.tensor(0.0),
            "kl": kl_loss,
        }

class VAETrainer():
    def __init__(self,
                 data_dir: str,
                 output_dir: str = "../checkpoint",
                 from_pretrain: str = "../model/v1-5-pruned-emaonly.safetensors",
                 frozen_weights: str = "",
                 image_size: int = 512,
                 batch_size: int = 8,
                 num_workers: int = 16,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 2,
                 save_every: int = 2,
                 lpips_weight: float = 0.1,
                 kl_weight: float = 1e-6,
                 scaling_factor: float = 1.0,
                 device='cuda',
                 dtype: torch.dtype = torch.float32):

        self.output_dir = output_dir
        self.image_size = image_size
        self.num_epochs = num_epochs
        self.scaling_factor = scaling_factor
        self.save_every = save_every
        self.device = device
        self.dtype = dtype
        
        os.makedirs(output_dir, exist_ok=True)

        ##### 1. 初始化 encoder 和 decoder
        self.encoder = get_encoder(ckp_path=from_pretrain, device=device, dtype=dtype)
        self.decoder = get_decoder(ckp_path=from_pretrain, device=device, dtype=dtype)
        
        print("="*80)

        ##### 2. 冻结权重
        if frozen_weights == 'encoder':
            print("🔄 Frozen encoder...")
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif frozen_weights == 'decoder':
            print("🔄 Frozen decoder...")
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False
        print("> Finished!")

        print("="*80)
        
        ##### 3. 处理数据集
        print("📂 Loading dataset...")
        self.dataset = ImageTextDataset(data_dir=data_dir, image_size=image_size)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device=='cuda' else False,
            drop_last=True,
        )
        # 展示前三张图像
        fig, axs = plt.subplots(1, 3)
        for idx, ax in enumerate(axs):
            img = self.dataset[idx]['image']
            img = (img.permute(1, 2, 0) + 1) / 2
            ax.imshow(img)
        plt.show()

        print("="*80)

        ##### 4. 初始化损失、优化器及调度器等
        print("🔄 Initializing Loss, optimizer and scheduler...")
        self.criterion = VAELoss(device=device)
        self.optimizer = AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        self.scaler = GradScaler('cuda')
        self.epoch = 0
        self.ckp_epochs = 0
        print("> Finished!")
        
        print("="*80)

        ##### 5. 打印配置信息
        print("\n📊 Configuration:")
        print(f"   DATASET DIRECTORY:    {data_dir}")
        print(f"   DATASET SIZE:         {len(self.dataset)}")
        print(f"   CHECKPOINT DIRECTORY: {output_dir}")
        print(f"   PRETRAIN MODEL PATH:  {from_pretrain}")
        print(f"   IMAGE SIZE:           {image_size}x{image_size}")
        print(f"   BATCH SIZE:           {batch_size}")
        print(f"   NUM WORKERS:          {num_workers}")
        print(f"   LEARNING RATE:        {learning_rate}")
        print(f"   TOTAL EPOCHS:         {num_epochs}")
        print(f"   LPIPS WEIGHT:         {lpips_weight}")
        print(f"   KL WEIGHT:            {kl_weight}")
        print(f"   SAVE INTERVAL:        {save_every}")
        print(f"   DEVICE:               {device}")
        print(f"   DTYPE:                {dtype}")

        print("="*80)
        
        print("✅ **All initialization is ready, start training using trainer.train()**")


    def train_epoch(self):
        epoch_losses = {"total": 0, "recon": 0, "lpips": 0, "kl": 0}
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {self.epoch+self.ckp_epochs}/{self.num_epochs+self.ckp_epochs}")
        
        for batch in progress_bar:            
            ##### 1. 前向传播生成损失
            with autocast('cuda', dtype=torch.bfloat16):
                ##### Encode
                # images shape: (batch_size, 3, height, width)
                # noise shape:  (batch_size, 4, height // 8, width // 8)
                images = batch["image"].to(self.device, dtype=torch.bfloat16)
                noise = torch.randn(images.shape[0], 4, self.image_size // 8, self.image_size // 8, device=self.device, dtype=torch.bfloat16)
                latent = self.encoder(images, noise, self.scaling_factor)
                
                ##### Decode
                # (batch_size, 3, height, width)
                recon = self.decoder(latent, self.scaling_factor)
                
                ##### Loss
                # {"total": 0.4, "recon": 0.3, "lpips": 0.2, "kl": 0.1}
                losses = self.criterion(recon, images, self.encoder.mean, self.encoder.log_var)

            ##### 2. 反向传播并更新参数
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()),max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            ##### 3. 更新损失和进度条信息
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            progress_bar.set_postfix({
                "total": f"{losses['total'].item():.4f}",
                "recon": f"{losses['recon'].item():.4f}",
                "kl": f"{losses['kl'].item():.4f}",
            })

        return epoch_losses
                
    def train(self):
        print(f"🚀 Starting VAE training for {self.num_epochs} epochs...")
    
        for _ in range(self.num_epochs):
            self.epoch += 1

            # 恢复训练状态
            self.encoder.train()
            self.decoder.train()

            # 训练一个 Epoch
            epoch_losses = self.train_epoch()

            # 更新学习率调度
            self.scheduler.step()
            
            for key in epoch_losses:
                epoch_losses[key] /= len(self.dataloader)
            
            print(f"\n📊 Epoch {self.ckp_epochs + self.epoch} Summary:")
            print(f"   Total: {epoch_losses['total']:.6f}")
            print(f"   Recon: {epoch_losses['recon']:.6f}")
            print(f"   LPIPS: {epoch_losses['lpips']:.6f}")
            print(f"   KL:    {epoch_losses['kl']:.6f}")
            print(f"   LR:    {self.scheduler.get_last_lr()[0]:.2e}")

            # 可视化
            self.visualize_prediction()
    
            ##### 保存检查点
            if self.epoch % self.save_every == 0 or self.epoch == self.num_epochs:
                self.save_checkpoint()
    
        print("✅ VAE training complete!")

    def visualize_prediction(self):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                # 验证模式
                self.encoder.eval()
                self.decoder.eval()

                # 随机抽样图像
                idx = random.randint(0, len(self.dataset) - 1)
                input_img = self.dataset[idx]['image'].unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)

                # 编码并解码
                noise = torch.randn(1, 4, self.image_size // 8, self.image_size // 8, device=self.device, dtype=torch.bfloat16)
                output_img = self.decoder(self.encoder(input_img, noise))
                
                # 展示解码结果
                post_process = lambda img: ((img.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(-1, 1) + 1) / 2 * 255).byte()
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(post_process(input_img))
                axs[0].set_title('Input image')
                axs[1].imshow(post_process(output_img))
                axs[1].set_title('Rebuilt image')
                plt.show()
    
    def save_checkpoint(self, path = None, name = None):
        checkpoint = {
            "epoch": self.epoch+self.ckp_epochs,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None
        }
        save_path = os.path.join(
            path if path else self.output_dir, 
            name if name else f"vae_epoch_{self.ckp_epochs + self.epoch}.pt"
        )
        torch.save(checkpoint, save_path)
        
        print(f"💾 Checkpoint saved: {save_path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.ckp_epochs = checkpoint["epoch"]
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        
        print(f"📂 Checkpoint loaded: {path} (epoch {self.ckp_epochs})")