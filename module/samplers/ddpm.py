import torch
from torch import nn
import numpy as np
from .sampler import Sampler

class DDPMSampler(Sampler):
    """Denoising Diffusion Probabilistic Models Sampler"""

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ) -> None:
        
        super().__init__(generator, num_training_steps, beta_start, beta_end)

    def step(
        self, 
        t: int, 
        latents: torch.Tensor, 
        model_output: torch.Tensor, 
        noise: torch.Tensor | None = None
    ) -> torch.FloatTensor:
        
        #### 1. 参数
        # t-1
        prev_t = t - self.num_training_steps // self.num_inference_steps
        #prev_t = t - stride
        # ᾱₜ
        alpha_prod_t = self.alpha_cumprod[t]
        # ᾱₜ₋₁
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, dtype=torch.float32)
        # αₜ
        alpha_t = alpha_prod_t / alpha_prod_t_prev

        ##### 2. 计算均值、方差、标准差
        # [xₜ-[(1-αₜ)/√(1-ᾱₜ)]ϵ]/√αₜ
        mean = (latents - ((1 - alpha_t) / (1 - alpha_prod_t).sqrt()) * model_output) / alpha_t.sqrt()
        # (1-αₜ)(1-ᾱₜ₋₁)/(1-ᾱₜ)
        variance = (1 - alpha_t) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        # Avoid division by zero
        std = torch.clamp(variance, min=1e-20).sqrt()

        ##### 3. 从对应的高斯分布中生成样本
        if t > 0:
            # Samplize from N(0, I)
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            ) if noise is None else noise
            # N(0, 1) --> N(mu, sigma^2)
            # X = mu + sigma * Z where Z ~ N(0, 1)
            sample = mean + std * noise
        else:
            sample = mean + torch.tensor(0.0, dtype=latents.dtype, device=latents.device)

        return sample