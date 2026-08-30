import torch
import numpy as np
from .ddpm import DDPMSampler

class DDIMSampler(DDPMSampler):
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        ddim_eta: float = 0.0,
    ) -> None:
        super().__init__(generator, num_training_steps, beta_start, beta_end)
        
        self.ddim_eta = ddim_eta

    def step(
        self, t: int, latents: torch.Tensor, model_output: torch.Tensor
    ) -> torch.FloatTensor:
        """Perform a single step of the DDPM sampler."""
        
        # t-1
        prev_t = t - self.num_training_steps // self.num_inference_steps
        # ᾱₜ
        alpha_prod_t = self.alpha_cumprod[t]
        # ᾱₜ₋₁
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, dtype=torch.float32)
        # αₜ
        alpha_t = alpha_prod_t / alpha_prod_t_prev

        ##### 2. Mean, variance and std
        variance = self.ddim_eta ** 2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_t)
        std = torch.clamp(variance, min=0.0).sqrt()
        
        pred_original_sample = (latents - (1 - alpha_prod_t).sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_original_sample_coeff = alpha_prod_t_prev.sqrt()
        current_sample_coeff = torch.sqrt(1 - alpha_prod_t_prev - variance)
        
        ##### 3. Samplize
        if t > 0 and variance > 0.0:
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * model_output + std * noise
        else:
            sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * model_output + torch.tensor(0.0, dtype=latents.dtype, device=latents.device)
        
        return sample