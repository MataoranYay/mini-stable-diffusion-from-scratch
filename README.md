# Mini Stable Diffusion from Scratch

一个基于 PyTorch 从零实现的 **Stable Diffusion v1.x** 教学/研究项目，包含完整的 VAE、CLIP 文本编码器、UNet 扩散模型、DDPM/DDIM 采样器，以及可交互的 Gradio WebUI，适合学习扩散模型架构或进行小规模微调实验。

> **声明**：本项目为教育性质实现，核心模块均从零手写，不依赖 `diffusers`、`stable-diffusion-webui` 等现成框架，便于理解 SD 的每个组件。

---

## 目录

- [Mini Stable Diffusion from Scratch](#mini-stable-diffusion-from-scratch)
  - [目录](#目录)
  - [核心特性](#核心特性)
  - [快速开始](#快速开始)
  - [项目结构](#项目结构)
  - [环境要求](#环境要求)
  - [安装](#安装)
  - [数据集准备](#数据集准备)
  - [训练](#训练)
    - [1. VAE 训练](#1-vae-训练)
    - [2. Diffusion (UNet) 训练](#2-diffusion-unet-训练)
  - [推理](#推理)
    - [通过 Jupyter Notebook](#通过-jupyter-notebook)
    - [通过 Gradio WebUI](#通过-gradio-webui)
  - [模型架构](#模型架构)
    - [VAE](#vae)
    - [CLIP](#clip)
    - [UNet Diffusion](#unet-diffusion)
    - [采样器](#采样器)
  - [训练记录](#训练记录)
    - [训练环境](#训练环境)
    - [VAE 训练结果](#vae-训练结果)
    - [Diffusion 训练结果](#diffusion-训练结果)
  - [生成示例](#生成示例)
  - [注意事项与限制](#注意事项与限制)
  - [致谢](#致谢)
  - [许可证](#许可证)

---

## 核心特性

- **从零实现核心组件**：VAE Encoder/Decoder、CLIP Text Encoder、UNet 扩散模型、Self/Cross Attention、Time Embedding。
- **完整训练流程**：支持分阶段训练 VAE 与 Diffusion UNet，支持从 SD 官方权重初始化。
- **CFG 训练支持**：Diffusion 训练阶段以 10% 概率随机丢弃文本提示词，使推理时可以使用 Classifier-Free Guidance（CFG）。
- **多种采样器**：内置 DDPM 与 DDIM 采样器，支持文生图与图生图（image-to-image）。
- **两种交互方式**：
  - `train.ipynb`：Jupyter Notebook 训练与可视化调试。
  - `webui.py`：基于 Gradio 的图形化生成界面。
- **权重加载**：支持直接加载官方 `.safetensors` 权重，或加载自行训练的检查点。

---

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/your-username/mini-stable-diffusion.git
cd mini-stable-diffusion

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# 3. 下载 SD v1.5 基础权重到 model/ 目录
# 如：model/base-v1-5-pruned-emaonly.safetensors

# 4. 启动 WebUI
python webui.py
```

---

## 项目结构

```
mini-stable-diffusion/
│
├── checkpoint/                 # 训练保存的检查点
│   ├── vae_epoch_1.pt          # VAE 检查点
│   └── diffusion_epoch_2.pt    # Diffusion 检查点
│
├── dataset/                    # 数据集目录
│   └── minecraft-preview/      # 示例数据集
│       ├── images/             # 训练图像
│       └── captions.json       # 图像-文本标注
│
├── module/                     # 核心模型模块
│   ├── attention.py            # Self / Cross Attention
│   ├── clip.py                 # CLIP 文本编码器
│   ├── diffusion.py            # UNet 扩散模型
│   ├── model_loader.py         # 权重加载工具
│   ├── pipeline.py             # 文生图 / 图生图推理管道
│   ├── vae.py                  # VAE Encoder / Decoder
│   └── samplers/               # 采样器
│       ├── sampler.py          # 基础采样器（加噪 / 去噪）
│       ├── ddpm.py             # DDPM 采样器
│       └── ddim.py             # DDIM 采样器
│
├── tokenizer/                  # CLIP tokenizer 文件
│   ├── vocab.json
│   └── merges.txt
│
├── trainer/                    # 训练器
│   ├── dataset.py              # ImageTextDataset
│   ├── train_vae.py            # VAE 训练器
│   └── train_diffusion.py      # Diffusion 训练器
│
├── train.ipynb                 # 训练与推理 Notebook
├── webui.py                    # Gradio WebUI 入口
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

---

## 环境要求

| 项目 | 推荐配置 |
|------|---------|
| 操作系统 | Windows 10/11 / Linux |
| Python | 3.10 ~ 3.12 |
| PyTorch | 2.12.1 |
| CUDA | 建议12.x 或更高 |
| GPU 显存 | 训练 512×512 建议 ≥ 16 GB；推理 512×512 建议 ≥ 8 GB |
| 训练显卡 | NVIDIA GeForce RTX 5090 |

---

## 安装

1. 创建并激活虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

```text
gradio==6.26.0
ipython==8.12.3
lpips==0.1.4
matplotlib==3.11.1
numpy==2.5.2
Pillow==12.3.0
safetensors==0.8.0
torch==2.12.1
torchvision==0.28.0
tqdm==4.70.0
transformers==5.12.1
```

3. 下载 Stable Diffusion v1.5 官方基础权重（如 `base-v1-5-pruned-emaonly.safetensors`）并放置到 `model/` 目录下，用于初始化 CLIP、VAE 与 Diffusion。

---

## 数据集准备

项目使用 **图像-文本对** 数据集。期望的目录结构如下：

```
dataset/your-dataset/
├── images/
│   ├── 00001.jpg
│   ├── 00002.png
│   └── ...
└── captions.json
```

其中 `captions.json` 格式为：

```json
{
    "00001.jpg": "a photo of a minecraft creeper",
    "00002.png": "a male wearing 3D glasses",
    "...": "..."
}
```

也支持 `metadata.csv` 格式（`file_name,text`），或在没有标注文件时自动使用文件名作为 caption。

示例数据集 `https://huggingface.co/monadical-labs/minecraft-preview` 包含 1000 张 Minecraft 风格图像及对应英文描述。

---

## 训练

训练分为两个阶段：**VAE 自编码器** 与 **Diffusion UNet**。VAE 负责将图像压缩到 latent 空间，Diffusion 负责在 latent 空间中学习去噪。

### 1. VAE 训练

```python
from trainer.train_vae import VAETrainer

trainer = VAETrainer(
    data_dir='dataset/minecraft-preview/',
    output_dir='checkpoint/',
    from_pretrain='model/base-v1-5-pruned-emaonly.safetensors',
    frozen_weights='encoder',      # 可冻结 encoder 只训练 decoder
    image_size=512,
    batch_size=4,
    num_workers=16,
    learning_rate=1e-4,
    num_epochs=1,
    lpips_weight=0.1,
    kl_weight=1e-6,
    scaling_factor=1.0,
    save_every=1,
    device='cuda',
)

trainer.train()
```

**损失函数**：

```text
Loss = L1_recon + MSE_recon + λ_lpips * LPIPS + λ_kl * KL
```

- `L1_recon` / `MSE_recon`：像素级重建误差
- `LPIPS`：感知损失（VGG 网络）
- `KL`：latent 分布与标准正态分布的 KL 散度

### 2. Diffusion (UNet) 训练

```python
from trainer.train_diffusion import DiffusionTrainer

diffusion_trainer = DiffusionTrainer(
    data_dir='dataset/minecraft-preview/',
    output_dir='checkpoint/',
    vae_ckp='checkpoint/vae_epoch_1.pt',
    clip_ckp='model/base-v1-5-pruned-emaonly.safetensors',
    diffusion_ckp='model/base-v1-5-pruned-emaonly.safetensors',
    tokenizer_dir='tokenizer',
    image_size=512,
    batch_size=8,
    num_workers=16,
    learning_rate=1e-5,
    num_epochs=2,
    save_every=2,
    prompt_dropout=0.1,            # 10% 概率丢弃文本，用于 CFG
    device='cuda',
    dtype=torch.float32,
    scaling_factor=0.18215,
)

diffusion_trainer.train()
```

**训练目标**：

```text
L = || ε - ε_θ(x_t, t, c) ||²
```

其中 `x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε`，`c` 为 CLIP 文本条件。通过最小化 MSE，使 UNet 学会根据文本提示预测并去除噪声。

---

## 推理

### 通过 Jupyter Notebook

打开 `train.ipynb`，按顺序执行 **Diffusion Training → Eval** 单元格即可加载检查点并生成图像。

```python
from transformers import CLIPTokenizer
from module import model_loader, pipeline

DEVICE = 'cuda'
DTYPE = torch.bfloat16

tokenizer = CLIPTokenizer('tokenizer/vocab.json', 'tokenizer/merges.txt')
models = {
    'encoder':   model_loader.get_model('encoder',   'checkpoint/vae_epoch_1.pt', device=DEVICE, dtype=DTYPE).eval(),
    'decoder':   model_loader.get_model('decoder',   'checkpoint/vae_epoch_1.pt', device=DEVICE, dtype=DTYPE).eval(),
    'clip':      model_loader.get_model('clip',      'model/base-v1-5-pruned-emaonly.safetensors', device=DEVICE, dtype=DTYPE).eval(),
    'diffusion': model_loader.get_model('diffusion', 'checkpoint/diffusion_epoch_2.pt', device=DEVICE, dtype=DTYPE).eval(),
}

generator = pipeline.generate(
    prompt='A male wearing 3D glasses.',
    uncond_prompt='',
    img_width=512,
    img_height=512,
    do_cfg=True,
    cfg_scale=8.5,
    sampler_name='ddpm',
    n_inference_steps=64,
    num_training_steps=1000,
    models=models,
    seed=42,
    device=DEVICE,
    dtype=DTYPE,
    tokenizer=tokenizer,
    decode_interval=6,
)

for output_image in generator:
    display(Image.fromarray(output_image[0]))
```

### 通过 Gradio WebUI

```bash
python webui.py
```

浏览器将自动打开交互式界面，支持：

- 正向 / 负向提示词
- 图像分辨率调节（64 的倍数）
- DDPM / DDIM 采样器切换
- CFG 强度与推理步数
- 图生图（上传参考图并设置 strength）
- 随机种子控制

---

## 模型架构

### VAE

- 编码器将图像 `3×H×W` 压缩为 latent `4×(H/8)×(W/8)`。
- 解码器将 latent 重建回图像。
- 训练时使用重参数化技巧采样 latent，并通过 `scaling_factor`（默认 0.18215）缩放。

### CLIP

- 使用本地 `tokenizer/` 下的 `vocab.json` 与 `merges.txt`。
- 文本编码为 `77` 个 token，输出 `77×768` 的文本嵌入，作为 UNet 的 cross-attention 条件。

### UNet Diffusion

- 时间步通过正弦位置编码（sinusoidal time embedding）映射为 `1×320`。
- UNet 包含 Encoder、Bottleneck、Decoder，配合 Residual Block、Self-Attention、Cross-Attention 与 UpSample 层。
- 输入为带噪 latent，输出为预测的噪声。

### 采样器

- **DDPM**：逐步去噪，每个时间步都注入随机噪声。
- **DDIM**：确定性采样，可通过 `ddim_eta` 控制随机性。
- 图生图时通过 `strength` 控制从哪一步开始去噪。

---

## 训练记录

### 训练环境

| 配置项 | 值 |
|--------|-----|
| GPU | NVIDIA GeForce RTX 5090 |
| 框架 | PyTorch 2.12.1 |
| CUDA | 12.x |
| 数据类型 | `bfloat16`（推理）/ `float32`（Diffusion 训练） |
| 数据集 | `dataset/minecraft-preview/`（1000 张图像-文本对） |

### VAE 训练结果

| 参数 | 值 |
|------|-----|
| 图像尺寸 | 512 × 512 |
| Latent 尺寸 | 64 × 64 |
| 批次大小 | 4 |
| Workers | 16 |
| 学习率 | 1e-4 |
| 训练轮数 | 1 |
| LPIPS 权重 | 0.1 |
| KL 权重 | 1e-6 |
| 每轮步数 | 250 |
| 单轮耗时 | ~1 分 41 秒（~2.46 it/s） |

**Epoch 1 损失**：

```text
Total: 0.536620
Recon: 0.016488
LPIPS: 0.022892
KL:    517842.616125
LR:    1.00e-06
```

**保存检查点**：`checkpoint/vae_epoch_1.pt`

### Diffusion 训练结果

| 参数 | 值 |
|------|-----|
| 图像尺寸 | 512 × 512 |
| Latent 尺寸 | 64 × 64 |
| 批次大小 | 8 |
| Workers | 16 |
| 学习率 | 1e-5 |
| 训练轮数 | 2 |
| DDPM 训练步数 | 1000 |
| Prompt Dropout | 0.1（10%） |
| 每轮步数 | 125 |
| 单轮耗时 | ~50 秒（~2.5 it/s） |

**损失曲线**：

| Epoch | Average Loss | Learning Rate |
|-------|--------------|---------------|
| 1 | 0.057893 | 5.05e-06 |
| 2 | 0.051749 | 1.00e-07 |

**保存检查点**：`checkpoint/diffusion_epoch_2.pt`

---

## 生成示例

使用以下参数生成的示例：

```python
prompt='A male wearing 3D glasses.'
uncond_prompt=''
img_width=512
img_height=512
do_cfg=True
cfg_scale=8.5
sampler_name='ddpm'
n_inference_steps=64
num_training_steps=1000
```

> 受数据集规模与训练轮数限制，生成结果会带有明显的 Minecraft 数据集风格偏差。增加数据量与训练轮数可进一步改善提示词遵循度与图像质量。

---

## 注意事项与限制

1. **分辨率一致性**：训练与推理的分辨率应保持一致。若训练使用 512×512，推理也请使用 512×512（或其倍数），否则 UNet 会 out-of-distribution。
2. **CFG 需要训练支持**：若希望推理时 CFG 效果明显，训练阶段必须启用 `prompt_dropout`（建议 0.1）。
3. **数据集规模**：本项目仅使用 1000 张图像进行演示，远小于生产级 Stable Diffusion 模型。生成结果主要用于验证代码正确性。
4. **VAE 与 Diffusion 分阶段训练**：必须先训练/准备好 VAE 检查点，再训练 Diffusion。

---

## 致谢

- 模型架构与权重格式参考 [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
- CLIP tokenizer 文件来自 OpenAI CLIP。
- 数据集 https://huggingface.co/monadical-labs/minecraft-preview

---

## 许可证

本项目仅用于学习与研究目的。生成的模型权重与代码遵循各自原始许可证（如 SD 权重受其原许可协议约束）。项目代码部分可基于 MIT 许可证使用，请保留原作者声明。
