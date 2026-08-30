import torch
import numpy as np
from tqdm import tqdm
from module.samplers.ddpm import DDPMSampler
from module.samplers.ddim import DDIMSampler

def generate(
    prompt: str,                             # 正向提示词
    uncond_prompt: str,                      # 负向提示词
    img_width: int = 512,                    # 生成图像的宽度
    img_height: int = 512,                   # 生成图像的高度
    input_image: torch.Tensor | None = None, # 可选的输入图像参考
    strength: float = 0.8,                   # 输入图像的加噪强度
    do_cfg: bool = True,                     # 是否使用分类器自由引导（CFG）
    cfg_scale: float = 7.5,                  # CFG 引导强度
    sampler_name: str = "ddpm",              # 采样器名称
    n_inference_steps: int = 50,             # 推理步数
    num_training_steps: int = 1000,          # 总训练步数
    models: dict[str, torch.nn.Module] | None = None, # 模型字典
    seed: int | None = None,                 # 生成随机种子
    device: str | None = None,               # 运行模型的设备
    dtype: torch.dtype = torch.bfloat16,     # 数据类型精度
    tokenizer = None,                        # 分词器
    ddim_eta: float = 0.0,                   # DDIM 采样中的 eta 参数
    decode_interval: int = 1,                # 解码中间 latent 的间隔步数
    scaling_factor: float = 0.18215,         # Stable Diffusion VAE 的缩放因子
) -> torch.Tensor:
    
    with torch.inference_mode():
        # =============================== 1. 初始化随机数种子 ===============================
        generator = torch.Generator(device=device)
        generator.seed() if seed is None else generator.manual_seed(seed)

        # =============================== 2. CLIP 模型：处理提示词 ===============================
        # 获取 CLIP 模型
        clip = models["clip"]
        clip = clip.to(device)
        # 是否启用 CFG 引导
        if do_cfg:
            # 对正向和负向提示词进行分词
            # 形状: (1, 77)
            cond_tokens = tokenizer([prompt], padding="max_length", max_length=77, truncation=True).input_ids
            uncond_tokens = tokenizer([uncond_prompt], padding="max_length", max_length=77, truncation=True).input_ids
            
            # 转换为张量
            # 形状: (1, 77)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            uncond_tokens = torch.tensor(uncond_tokens,dtype=torch.long,device=device)

            # 送入 CLIP 模型
            # 形状: (1, 77) -> (1, 77, 768)
            cond_context = clip(cond_tokens)
            uncond_context = clip(uncond_tokens)

            # 在第 0 维拼接条件上下文与非条件上下文
            # 形状: (2, 77, 768)
            context = torch.cat([cond_context, uncond_context], dim=0)
        else:
            # 对正向提示词进行分词
            # 形状: (1, 77)
            cond_tokens = tokenizer([prompt], padding="max_length", max_length=77, truncation=True).input_ids

            # 转换为张量
            # 形状: (1, 77)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # 送入 CLIP 模型
            # 形状: (1, 77) -> (1, 77, 768)
            context = clip(cond_tokens)

        # =============================== 3. 采样器 ===============================
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator, num_training_steps)
            sampler.set_inference_steps(n_inference_steps)
        elif sampler_name == "ddim":
            sampler = DDIMSampler(generator, num_training_steps, ddim_eta=ddim_eta)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")


        # =============================== 4. VAE 编码器：处理输入图像参考 ===============================
        # VAE 编码器输出的形状
        latents_shape = (1, 4, img_height // 8, img_width // 8)
        # 是否根据输入图像引导生成
        if input_image is not None:
            # 获取 VAE 编码器
            encoder = models["encoder"]
            encoder = encoder.to(device)

            # 预处理输入图像
            # 形状: (1, 3, img_width, img_height)
            input_image = input_image.resize((img_width, img_height))
            input_image = torch.tensor(np.array(input_image), dtype=dtype, device=device)
            input_image = rescale(input_image, (0, 255), (-1, 1))
            input_image = input_image.unsqueeze(0).permute(0, 3, 1, 2)

            # 图像通过 VAE 编码器
            encoder_noise = torch.randn(latents_shape, dtype=dtype, device=device, generator=generator)
            latents = encoder(input_image, encoder_noise, scaling_factor)

            # 按指定强度对编码器输出的 latent 加噪
            sampler.set_strength(strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

        else:
            # 文生图：从标准正态分布 N(0, I) 中采样随机噪声
            # 形状: (1, 4, img_height // 8, img_width // 8)
            latents = torch.randn(latents_shape, dtype=dtype, device=device, generator=generator)

        # =============================== 5. 扩散模型 / VAE 解码器：预测噪声并去噪 ===============================
        # 获取扩散模型
        diffusion = models["diffusion"]
        diffusion = diffusion.to(device)
        
        # 获取 VAE 解码器
        decoder = models["decoder"]
        decoder = decoder.to(device)
        
        # 推理总步数 / 解码间隔
        timesteps = tqdm(sampler.timesteps, desc="Sampling", unit="step")
        # 保证触发 decode_interval 次 yield，且最后一次循环一定触发
        trigger_indices = {len(timesteps) - 1} if (decode_interval == 1) else {round((len(timesteps) - 1) * i / (decode_interval - 1)) for i in range(decode_interval)}

        #生成循环
        for i, t in enumerate(timesteps):
            ##### 时间步嵌入
            time_embedding = get_time_embedding(t, device=device, dtype=dtype)

            ##### 预测噪声
            # 克隆 latent 以避免数据污染
            model_input = latents.detach().clone()
            if do_cfg:
                # 形状: (1, 4, img_height // 8, img_width // 8) -> (2, 4, img_height // 8, img_width // 8)
                model_input = model_input.repeat(2, 1, 1, 1)

                # 扩散模型预测噪声
                # 形状: (2, 4, img_height // 8, img_width // 8)
                model_output = diffusion(model_input, context, time_embedding)

                # 拆分为正向和负向特征
                # 形状: (1, 4, img_height // 8, img_width // 8)
                output_cond, output_uncond = model_output.chunk(2, dim=0)
                
                # 加权求和：cfg * cond + (1 - cfg) * uncond
                # 形状: (1, 4, img_height // 8, img_width // 8)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            else:
                # 扩散模型预测噪声
                # 形状: (1, 4, img_height // 8, img_width // 8)
                model_output = diffusion(model_input, context, time_embedding)

            ##### 去噪
            # 形状: (1, 4, img_height // 8, img_width // 8)
            latents = sampler.step(t, latents, model_output)

            ##### 解码 latent
            if i in trigger_indices:        
                # 通过 VAE 解码器解码 latent
                # 形状: (1, 4, img_height // 8, img_width // 8) -> (1, 3, img_height, img_width)
                output_image = decoder(latents.detach().clone(), scaling_factor)
        
                # 反归一化 / 通道顺序转换
                # 形状: (1, 3, img_height, img_width) -> (1, img_height, img_width, 3)
                output_image = rescale(output_image)
                output_image = output_image.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

                yield output_image

def rescale(
    tensor: torch.Tensor,
    old_range: tuple[float, float] = (-1, 1),
    new_range: tuple[float, float] = (0, 255),
    clamp: bool = True,
) -> torch.Tensor:
    """将张量从 old_range 缩放到 new_range。"""

    # 获取极值
    old_min, old_max = old_range
    new_min, new_max = new_range

    # 将张量缩放到 [0, 1]
    scaled_tensor = (tensor - old_min) / (old_max - old_min)

    # 缩放到新的范围
    rescaled_tensor = scaled_tensor * (new_max - new_min) + new_min

    # 截断数值
    if clamp: rescaled_tensor = rescaled_tensor.clamp(new_min, new_max)

    return rescaled_tensor

def get_time_embedding(timestep: torch.Tensor, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    r"""
    获取给定时间步 t 的时间嵌入。
    
    $ emb_{pos}^{(i)} = \frac{pos}{10000^{i/d_{model}}} = pos \cdot \exp\!\left( -\frac{\ln 10000}{d_{model}} i \right),\ i \in \{0,1,\dots,\tfrac{d_{model}}{2}\},\ d_{model}=320 $
    
    $ pe_{pos} = \begin{bmatrix} \cos(emb_{pos}^{(0)}) & \cdots & \cos(emb_{pos}^{(d_{model}/2)}) & \sin(emb_{pos}^{(0)}) & \cdots & \sin(emb_{pos}^{(d_{model}/2)}) \end{bmatrix} $
    """
    half_dim = 160

    # 形状: (1, 160)
    emb = torch.exp(
        -np.log(10000) * torch.arange(0, half_dim, dtype=dtype) / half_dim
    ).to(device=device, dtype=dtype)
    
    emb = torch.tensor([timestep], device=device, dtype=dtype)[:, None] * emb[None, :]
    # 形状: (1, 320)
    pe = torch.cat([emb.cos(), emb.sin()], dim=-1)
    
    return pe
