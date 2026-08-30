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

from trainer.dataset import ImageTextDataset
from module.vae import VAE_Encoder, VAE_Decoder
from module.model_loader import get_model

import warnings;warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

class VAELoss(nn.Module):
    """
    Combined loss for VAE training.

    Loss = L1_recon + MSE_recon + λ_lpips * LPIPS + λ_kl * KL
    """
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
        ##### 1. Reconstruction losses
        l1_loss = F.l1_loss(recon_x, x)
        mse_loss = F.mse_loss(recon_x, x)
        recon_loss = l1_loss + mse_loss

        ##### 2. Perceptual loss
        lpips_loss = self.lpips(recon_x, x).mean() if self.lpips_weight > 0 else torch.tensor(0.0, device=device)

        ##### 3. KL Divergence: D_KL(q(z|x) || N(0,1))
        # For Gaussian: KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kl_loss = kl_loss / x.shape[0]  # Average over batch

        ##### 4. Sum of all losses
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
                 output_dir: str = "../checkpoint/vae",
                 from_pretrain: str = "../model/base-v1-5-pruned-emaonly.safetensors",
                 frozen_weights: str = "",
                 image_size: int = 512,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 100,
                 lpips_weight: float = 0.1,
                 kl_weight: float = 1e-6,
                 scaling_factor: float = 1.0,
                 save_every: int = 5,
                 device='cuda',
                 dtype: torch.dtype = torch.float32):

        self.output_dir = output_dir
        self.image_size = image_size
        self.num_epochs = num_epochs
        self.scaling_factor = scaling_factor
        self.save_every = save_every
        self.device = device
        os.makedirs(output_dir, exist_ok=True)

        ##### 1. Initialize models
        print("🔄 Initializing VAE_Encoder and VAE_Decoder...")
        self.encoder = VAE_Encoder().to(device)
        self.decoder = VAE_Decoder().to(device)
        print("✅ Initialization finished!")
        print("="*80)

        ##### 2. Load from pretrain
        self.encoder = get_model("encoder", from_pretrain, device=device, dtype=dtype)
        self.decoder = get_model("decoder", from_pretrain, device=device, dtype=dtype)
        print("="*80)

        ##### 3. Frozen weights
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
        
        ##### 4. Process datasets
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
        
        ##### 5. Show the first three pictures 
        fig, axs = plt.subplots(1, 3)
        for idx, ax in enumerate(axs):
            img = self.dataset[idx]['image']
            img = (img.permute(1, 2, 0) + 1) / 2
            ax.imshow(img)
        plt.show()
        # Show the corresponding captions
        for idx in range(3):
            print(f"Caption {idx+1}: {self.dataset[idx]['caption']}")
        print("="*80)

        ##### 6. Loss, optimizer and scheduler
        print("🔄 Initializing Loss, optimizer and scheduler...")
        self.criterion = VAELoss(device=device)
        self.optimizer = AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        # Mixed precision training
        self.scaler = GradScaler('cuda')
        self.epoch = 0
        self.ckp_epochs = 0
        print("✅ Initialization finished!")
        print("="*80)

        ##### 7. Summary
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

        print("="*80)
        print("✅ **All initialization is ready, start training using trainer.train()**")


    def train_epoch(self):
        epoch_losses = {"total": 0, "recon": 0, "lpips": 0, "kl": 0}
        progress_bar = tqdm(
            self.dataloader, 
            desc=f"Epoch {self.epoch+self.ckp_epochs}/{self.num_epochs+self.ckp_epochs}"
        )
        
        for batch in progress_bar:     
            # (batch_size, 3, height, width)
            images = batch["image"].to(self.device)

            ##### Sample noise for reparameterization
            noise = torch.randn(
                images.shape[0], 4, self.image_size // 8, self.image_size // 8,
                device=self.device
            )

            self.optimizer.zero_grad()
            
            ##### Forward pass with mixed precision
            with autocast('cuda'):
                # Encode
                latent = self.encoder(images, noise, self.scaling_factor)  # (B, 4, H/8, W/8)
                # Decode
                recon = self.decoder(latent, self.scaling_factor)  # (B, 3, H, W)
                # Compute loss
                losses = self.criterion(recon, images, self.encoder.mean, self.encoder.log_var)

            ##### Backward pass
            self.scaler.scale(losses["total"]).backward()

            ##### Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            ##### Update metrics
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            ##### Update progress bar
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

            ##### Train epoch
            self.encoder.train()
            self.decoder.train()
            epoch_losses = self.train_epoch()
            self.scheduler.step()
            
            ##### Epoch summary
            for key in epoch_losses:
                epoch_losses[key] /= len(self.dataloader)
    
            print(f"\n📊 Epoch {self.ckp_epochs + self.epoch} Summary:")
            print(f"   Total: {epoch_losses['total']:.6f}")
            print(f"   Recon: {epoch_losses['recon']:.6f}")
            print(f"   LPIPS: {epoch_losses['lpips']:.6f}")
            print(f"   KL:    {epoch_losses['kl']:.6f}")
            print(f"   LR:    {self.scheduler.get_last_lr()[0]:.2e}")

            ##### Randomly sample an image to visualize the reconstruction
            with torch.no_grad():
                self.encoder.eval()
                self.decoder.eval()
                
                idx = random.randint(0, len(self.dataset) - 1)
                noise = torch.randn(
                    1, 4, self.image_size // 8, self.image_size // 8,
                    device=self.device
                )
                input_img = self.dataset[idx]['image'].unsqueeze(0).to(self.device)
                output_img = self.decoder(self.encoder(input_img, noise))
                
                # Show input image and output image
                post_process = lambda img: ((img.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(-1, 1) + 1) / 2 * 255).byte()
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(post_process(input_img))
                axs[0].set_title('Input image')
                axs[1].imshow(post_process(output_img))
                axs[1].set_title('Rebuilt image')
                plt.show()
    
            ##### Save checkpoint
            if self.epoch % self.save_every == 0 or self.epoch == self.num_epochs:
                self.save_checkpoint()
    
        print("✅ VAE training complete!")

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

@torch.no_grad()
def compute_vae_scaling_factor(
    vae_encoder: VAE_Encoder,
    data_dir: str,
    image_size: int,
    save_path: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_batches: int = None,  # 用 None 表示整个数据集
) -> None:
    """计算 VAE Latent Scaling Factor，即全局标准差的倒数。"""
    
    vae_encoder.eval()
    vae_encoder.to(device=device, dtype=dtype)
    
    dataset = ImageTextDataset(data_dir=data_dir, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device=='cuda' else False,
        drop_last=False,
    )
    
    all_latents = []
    
    print(f"📊 Calculating latent distribution...")
    for i, batch in enumerate(tqdm(dataloader)):
        if num_batches is not None and i >= num_batches:
            break
            
        images = batch["image"].to(device=device, dtype=dtype)  # (B, 3, H, W) in [-1, 1]
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        
        # 采样噪声进行重参数化
        noise = torch.randn(B, 4, H // 8, W // 8, device=device, dtype=dtype)
        
        # Encoder 输出 (注意：这里不乘以 0.18215，取原始 latent)
        latent = vae_encoder(images, noise)  # (B, 4, H/8, W/8)
        
        # 如果 encoder 内部已经做了缩放，需要还原
        # latent = latent / 0.18215  # 取消已有的缩放
        
        all_latents.append(latent.cpu())
    
    # 合并所有 latent
    all_latents = torch.cat(all_latents, dim=0)  # (N, 4, H, W)
    
    # 计算全局统计量
    mean = all_latents.mean().item()
    std = all_latents.std().item()
    var = all_latents.var().item()
    
    # 计算 scaling factor: 使缩放后方差接近 1
    scaling_factor = 1.0 / std
    
    print(f"\n📈 LATENT STATISTICAL RESUALTS:")
    print(f"   SAMPLE SIZE: {all_latents.shape[0]}")
    print(f"   MEAN:        {mean:.6f} (应接近 0)")
    print(f"   STD:         {std:.6f}")
    print(f"   VAR:         {var:.6f}")
    print(f"\n✅ Scaling Factor = 1/{std:.4f} = {scaling_factor:.6f}")


    # 4. 保存到配置文件
    config = {
        "scaling_factor": scaling_factor,
        "computed_from": data_dir,
        "image_size": image_size,
        "latent_std": 1.0 / scaling_factor,
    }
    torch.save(config, f"{save_path}/vae_scaling_config.pt")
    print(f"\n💾 Configeration saved: vae_scaling_config.pt")