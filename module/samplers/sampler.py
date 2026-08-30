import torch
from torch import nn
import numpy as np

class Sampler(nn.Module):
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ) -> None:
        super().__init__()

        # 外部传入的随机种子生成器，用于同一管理所有内部噪声
        self.generator = generator

        # 训练参数
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32)** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        # 训练总步数与步数列表
        self.num_training_steps: int = num_training_steps
        # [999, 998, 997, ..., 2, 1, 0]
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy()).long()
        
        # 推理总步数
        self.num_inference_steps: int

    def set_inference_steps(self, num_inference_steps: int = 64) -> None:
        """
        指定推理步数，计算对应的去噪步长，并按照该步长间隔从总训练步数中选取时间步序列。
        
        如：
            num_inference_steps = 20 -> stride = 1000 / 20 = 50
            -> timesteps = [999, 949, 899, ..., 149, 99, 49]
            去噪时，根据 t 时刻预测 t-50 时刻的噪声。
        """
        
        if num_inference_steps > self.num_training_steps:
            raise ValueError(f"num_inference_steps ({num_inference_steps}) cannot be greater than num_training_steps ({self.num_training_steps})")
            
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // num_inference_steps
        self.timesteps = self.timesteps[::step_ratio].to(self.generator.device)

    def set_strength(self, strength: float | int = 1.0) -> None:
        """
        指定起始步数（比例），用于图生图时控制输入图像的加噪步数。
        如：
            strength = 0.8 -> start_step = 200
            -> timesteps = [799, 798, 797, ..., 2, 1, 0]
        """

        start_step = int((1 - strength) * self.num_inference_steps)
        self.timesteps = self.timesteps[start_step:]
            

    def add_noise(self, original_samples: torch.Tensor, t: torch.IntTensor, noise: torch.Tensor | None = None) -> torch.FloatTensor:
        """
        给定时间步 t，根据加噪公式由 x₀ 一步得到 xₜ。
        
        xₜ = √ᾱₜx₀ + √(1-ᾱₜ)ϵ
        """

        # √ᾱₜ
        mean = self.alpha_cumprod[t.cpu()].sqrt().view(-1, 1, 1, 1).to(original_samples.device, original_samples.dtype)
        # √(1-ᾱₜ)
        std = (1 - self.alpha_cumprod[t.cpu()]).sqrt().view(-1, 1, 1, 1).to(original_samples.device, original_samples.dtype)
        # N(0, I)
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype
        ) if noise is None else noise
        # √ᾱₜx₀+√(1-ᾱₜ)ϵ        
        return mean * original_samples + std * noise

    def denoise(self, noisy_latent: torch.Tensor, t: torch.IntTensor, predicted_noise: torch.Tensor) -> torch.FloatTensor:
        """
        给定时间步 t，变形加噪公式由 xₜ 反向得到 x₀。
        
        x₀ = [xₜ - √(1-ᾱₜ)ϵ] / √ᾱₜ
        """

        # ᾱₜ
        alpha_bar_t = self.alpha_cumprod[t.cpu().item()].view(-1, 1, 1, 1).to(noisy_latent.device, noisy_latent.dtype)
        # √ᾱₜ
        sqrt_alpha_bar_t = alpha_bar_t.sqrt()
        # √(1-ᾱₜ)
        sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()
        # x₀ = [xₜ - √(1-ᾱₜ)ϵ] / √ᾱₜ
        return (noisy_latent - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t