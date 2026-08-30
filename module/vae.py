import torch
from torch import nn
from torch.nn import functional as F

class VAE_AttentionBlock(nn.Module):
    """
    VAE 注意力层：计算输入图像对自身的卷积注意力修正。
    Args:
        in_channels: 512
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, in_channels, height, width)
        b, c, h, w = x.shape
        res = x

        ##### 1. GroupNorm
        # (batch_size, in_channels, H, W)
        x = self.norm(x)

        ##### 2. 投影到 q, k, v 空间
        # (batch_size, in_channels, H, W)
        q, k, v = self.q(x), self.k(x), self.v(x)

        ##### 3. 将像素空间维度合并，同时将其前置
        # (batch_size, H*W, in_channels)
        q = q.view(b, c, h * w).transpose(-1, -2)
        k = k.view(b, c, h * w).transpose(-1, -2)
        v = v.view(b, c, h * w).transpose(-1, -2)

        ##### 4. 将图像像素作为 seq_len、通道作为 embed_dim 计算单头注意力修正
        # (batch_size, H*W, in_channels)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        ##### 5. 还原为输入形状
        # (batch_size, in_channels, H, W)
        x = x.transpose(-1, -2).view(b, c, h, w)

        ##### 6. 输出层投影
        # x shape: (batch_size, in_channels, H, W)
        x = self.proj_out(x)

        ##### 7. 残差连接
        # x shape: (batch_size, in_channels, H, W)
        x += res
        
        return x

class VAE_ResidualBlock(nn.Module):
    """VAE 残差块：输入图像传入2个卷积块，再进行残差连接。"""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 如果输出通道数与输入不一致，则需要将输入通过 nin_shortcut 层映射到输出维度再做残差连接
        self.nin_shortcut: nn.Identity | nn.Conv2d
        if in_channels == out_channels:
            self.nin_shortcut = nn.Identity()
        else:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, in_channels, height, width)
        res = x

        ##### 1. 卷积块1
        # (batch_size, out_channels, height, width)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        ##### 2. 卷积块2
        # (batch_size, out_channels, height, width)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        ##### 3. 残差连接
        # (batch_size, out_channels, height, width)
        x += self.nin_shortcut(res)
        
        return x

class VAE_Downsample(nn.Module):
    """VAE 下采样层：通过卷积核大小为 3，步长为 2 的卷积层完成下采样，使图像尺寸整除 2（向下取整），输出通道数不变。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 手动在图像右侧和底部填充，这样图像尺寸不论是奇数还是偶数，下采样宽高整除 2 时将严格向下取整
        x = F.pad(x, (0, 1, 0, 1))
        # (batch_size, channels, height, width) -> (batch_size, channels, height // 2, width // 2)
        x = self.conv(x)
        return x

class VAE_Upsample(nn.Module):
    """VAE 上采样层：通过线性临近插值使图像尺寸扩大 2 倍，再通过一个卷积层，输出通道数不变。"""
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, height, width) -> (batch_size, channels, height * 2, width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # (batch_size, channels, height * 2, width * 2) -> (batch_size, channels, height * 2, width * 2)
        x = self.conv(x)
        return x

class VAE_DownBlock(nn.Module):
    """VAE 下采样块：通过 2 个 VAE_ResidualBlock 层和一个可选的 VAE_Downsample 层。"""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        
        self.block = nn.ModuleList([
            VAE_ResidualBlock(in_channels, out_channels),
            VAE_ResidualBlock(out_channels, out_channels),
        ])
        self.downsample = VAE_Downsample(out_channels) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, in_channels, height, width) -> (batch_size, out_channels, height, width)
        for layer in self.block:
            x = layer(x)
        if self.downsample is not None:
            # (batch_size, out_channels, height, width) -> (batch_size, out_channels, height // 2, width // 2)
            x = self.downsample(x)
        return x

class VAE_UpBlock(nn.Module):
    """VAE 上采样块：通过 3 个 VAE_ResidualBlock 层和一个可选的 VAE_Upsample 层。"""
    
    def __init__(self, in_channels: int, out_channels: int, upsample: bool) -> None:
        super().__init__()
        
        self.block = nn.ModuleList([
            VAE_ResidualBlock(in_channels, out_channels),
            VAE_ResidualBlock(out_channels, out_channels),
            VAE_ResidualBlock(out_channels, out_channels),
        ])
        self.upsample = VAE_Upsample(out_channels) if upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, in_channels, height, width) -> (batch_size, out_channels, height, width)
        for layer in self.block:
            x = layer(x)
        if self.upsample is not None:
            # (batch_size, out_channels, height, width) -> (batch_size, out_channels, height * 2, width * 2)
            x = self.upsample(x)
        return x

class VAE_MidBlock(nn.Module):
    """VAE 瓶颈层：包括 2 个 VAE_ResidualBlock 层和 1 个VAE_AttentionBlock层。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        
        self.block_1 = VAE_ResidualBlock(channels, channels)
        self.attn_1 = VAE_AttentionBlock(channels)
        self.block_2 = VAE_ResidualBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, height, width)
        x = self.block_1(x)
        x = self.attn_1(x)
        x = self.block_2(x)
        return x

class VAE_Encoder(nn.Module):
    """VAE 编码器：编码输入图像产生其在潜空间分布的均值和方差，并通过重参数化采样返回潜空间特征图。"""

    def __init__(self) -> None:
        super().__init__()

        # (batch_size, 3, height, width) -> (batch_size, 128, height, width)
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.down = nn.ModuleList([
            # (batch_size, 128, height, width) -> (batch_size, 128, height // 2, width // 2)
            VAE_DownBlock(128, 128, downsample=True),
            # (batch_size, 128, height // 2, width // 2) -> (batch_size, 256, height // 4, width // 4)
            VAE_DownBlock(128, 256, downsample=True),
            # (batch_size, 256, height // 4, width // 4) -> (batch_size, 512, height // 8, width // 8)
            VAE_DownBlock(256, 512, downsample=True),
            # (batch_size, 512, height // 8, width // 8) -> (batch_size, 512, height // 8, width // 8)
            VAE_DownBlock(512, 512, downsample=False),
        ])

        # (batch_size, 512, height // 8, width // 8)
        self.mid = VAE_MidBlock(512)

        self.norm_out = nn.GroupNorm(32, 512)
        # (batch_size, 512, height // 8, width // 8) -> (batch_size, 8, height // 8,  width // 8)
        self.conv_out = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, noise: torch.Tensor, scaling_factor: float = 0.18215) -> torch.Tensor:
        ##### 1. 通道数对齐
        # (batch_size, 3, height, width) -> (batch_size, 128, height, width)
        x = self.conv_in(x)

        ##### 2. 下采样块+瓶颈层块
        # (batch_size, 128, height, width) -> (batch_size, 512, height // 8, width // 8)
        for block in self.down:
            x = block(x)
        x = self.mid(x)

        ##### 3. 通道数还原
        # (batch_size, 512, height // 8, width // 8) -> (batch_size, 8, height // 8, width // 8)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        x = self.quant_conv(x)

        ##### 4. 重参数化采样
        # 将输出的 x 以通道维度分割成两部分，分别代表潜空间分布的均值和对数方差
        # (batch_size, 8, height // 8, width // 8) -> two tensors of shape (batch_size, 4, height // 8, width // 8)
        self.mean, self.log_var = torch.chunk(x, 2, dim=1)
        # 计算标准差
        std = torch.clamp(self.log_var, -30, 20).exp().sqrt()
        # 采样
        # (batch_size, 4, height // 8, width // 8)
        x = self.mean + std * noise

        ##### 5. 以一个统计系数缩放，使潜空间分布近似于单位方差
        x *= scaling_factor

        return x

class VAE_Decoder(nn.Module):
    """VAE 解码器：将 VAE 编码器产生的潜空间特征图解码回图像。"""

    def __init__(self):
        super().__init__()

        # (batch_size, 4, height // 8, width // 8)
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1, padding=0)
        # (batch_size, 4, height // 8, width // 8) -> (batch_size, 512, height // 8, width // 8)
        self.conv_in = nn.Conv2d(4, 512, kernel_size=3, padding=1)
        # (batch_size, 512, height // 8, width // 8)
        self.mid = VAE_MidBlock(512)

        self.up = nn.ModuleList([
            # (batch_size, 256, height, width) -> (batch_size, 128, height, width)
            VAE_UpBlock(256, 128, upsample=False),
            # (batch_size, 512, height // 2, width // 2) -> (batch_size, 256, height, width)
            VAE_UpBlock(512, 256, upsample=True),
            # (batch_size, 512, height // 4, width // 4) -> (batch_size, 512, height // 2, width // 2)
            VAE_UpBlock(512, 512, upsample=True),
            # (batch_size, 512, height // 8, width // 8) -> (batch_size, 512, height // 4, width // 4)
            VAE_UpBlock(512, 512, upsample=True),
        ])

        self.norm_out = nn.GroupNorm(32, 128)
        # (batch_size, 128, height, width) -> (batch_size, 3, height, width)
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, scaling_factor: float = 0.18215) -> torch.Tensor:
        ##### 1. 缩放还原
        # (batch_size, 4, height // 8, width // 8)
        x /= scaling_factor

        ##### 2. 通道数对齐
        # (batch_size, 4, height // 8, width // 8) -> (batch_size, 512, height // 8, width // 8)
        x = self.post_quant_conv(x)
        x = self.conv_in(x)
        x = self.mid(x)

        ##### 3. 上采样块，按逆索引执行
        # (batch_size, 512, height // 8, width // 8) -> (batch_size, 128, height, width)
        for block in reversed(self.up):
            x = block(x)

        ##### 4. 通道数还原
        # (batch_size, 128, height, width) -> (batch_size, 3, height, width)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x