import torch
import torch.nn as nn
from torch.nn import functional as F

class TimeEmbedding(nn.Sequential):
    """时间步嵌入层：将时间步嵌入到高维。"""

    def __init__(self, embed_dim: int = 320) -> None:
        # (1, 320) -> (1, 1280)
        super().__init__(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, 4 * embed_dim),
        )

class UNetResidualBlock(nn.Module):
    """UNet 残差块：将输入特征图映射到输出维度，同时融合时间步信息。"""

    def __init__(self, in_channels: int, out_channels: int, n_time=1280) -> None:
        super().__init__()

        # (batch_size, in_channels, height, width) -> (batch_size, out_channels, height, width)
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        # (1, 1280) -> (1, out_channels)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_time, out_channels),
        )

        # (batch_size, out_channels, height, width)
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.skip_connection = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # feature shape: (batch_size, in_channels, height, width)
        # t shape:       (1, 1280)
        res = feature

        ##### 1. 处理特征图
        # (batch_size, in_channels, height, width) -> (batch_size, out_channels, height, width)
        feature = self.in_layers(feature)

        ##### 2. 处理时间嵌入
        # (1, 1280) -> (1, out_channels)
        t = self.emb_layers(t)

        ##### 3. 融合特征图与时间嵌入
        # (batch_size, out_channels, height, width)
        merged = feature + t[:, :, None, None]
        merged = self.out_layers(merged)

        ##### 4. 残差连接
        # (batch_size, out_channels, height, width)
        merged += self.skip_connection(res)
        
        return merged

class UNetSelfAttention(nn.Module):
    """UNet 自注意力层：将特征图合并后的像素维度当作句子序列、通道维度当作词嵌入计算对自身的注意力修正。"""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        in_proj_bias: bool = False,
        out_proj_bias: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=out_proj_bias),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:        
        # (batch_size, img_len, embed_dim) = (batch_size, H*W, C)
        b, _, _ = input_shape = x.shape

        ##### 1. 投影到 q, k, v 空间
        # (batch_size, img_len, embed_dim)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        ##### 2. 拆分成多头，同时将头维度前置
        # (batch_size, num_heads, img_len, head_dim)
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        ##### 3. 计算注意力修正
        # (batch_size, num_heads, img_len, head_dim)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal_mask)

        ##### 4. 还原为输入形状
        # (batch_size, img_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(input_shape)

        ##### 5. 输出层投影
        # (batch_size, img_len, embed_dim) = (batch_size, H*W, C)
        output = self.to_out(output)

        return output

class UNetCrossAttention(nn.Module):
    """UNet 交叉注意力层：计算特征图与提示词嵌入之间的交叉注意力修正。"""
    
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        context_dim: int = 768,
        in_proj_bias: bool = False,
        out_proj_bias: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.to_k = nn.Linear(context_dim, embed_dim, bias=in_proj_bias)
        self.to_v = nn.Linear(context_dim, embed_dim, bias=in_proj_bias)
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=out_proj_bias),
            nn.Dropout(0.0),
        )

    def forward(self, latent: torch.Tensor, context: torch.Tensor, causal_mask: bool = False):
        # latent shape:  (batch_size, img_len, embed_dim) = (batch_size, H*W, C)
        # context shape: (batch_size, seq_len, context_dim) = (batch_size, 77, 768)
        batch_size, _, _ = latent_shape = latent.shape

        ##### 1. 投影到 q, k, v 空间
        # (batch_size, img_len, embed_dim)
        q = self.to_q(latent)
        # (batch_size, seq_len, context_dim) -> (batch_size, seq_len, embed_dim)
        k = self.to_k(context)
        v = self.to_v(context)
        
        ##### 2. 拆分成多头，同时将头维度前置
        # (batch_size, num_heads, img_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        ##### 3. 计算注意力修正
        # (batch_size, num_heads, img_len, head_dim)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal_mask)

        ##### 4. 还原为输入形状
        # (batch_size, img_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(latent_shape)

        ##### 5. 输出层投影
        # (batch_size, img_len, embed_dim) = (batch_size, H*W, C)
        output = self.to_out(output)
        
        return output
        
class GEGLU(nn.Module):
    """GEGLU 激活函数：将输入进行投影，并将其分割产生值 x 和门控 gate 两部分。"""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, dim_in) -> (batch_size, seq_len, dim_out / 2)
        x, gate = self.proj(x).chunk(2, dim=-1)

        # (batch_size, seq_len, dim_out / 2)
        x = x * F.gelu(gate)
        
        return x

class UNetFeedForward(nn.Module):
    """前馈层：输入特征图经 GEGLU 激活后传入线性层映射回原形状。"""

    def __init__(self, embed_dim: int, mult: int = 4) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            GEGLU(embed_dim, embed_dim * mult * 2),
            nn.Dropout(0.0),
            nn.Linear(embed_dim * mult, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UNetTransformerBlock(nn.Module):
    """UNetTransformer 块：传入特征图和提示词嵌入，分别经过自注意力层、交叉注意力层、前馈层处理。"""

    def __init__(self, n_head: int, embed_dim: int, context_dim: int = 768) -> None:
        super().__init__()
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn1 = UNetSelfAttention(n_head, embed_dim, in_proj_bias=False)

        # Cross-Attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn2 = UNetCrossAttention(n_head, embed_dim, context_dim, in_proj_bias=False)

        # Feed-Forward
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = UNetFeedForward(embed_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x shape:       (batch_size, seq_len, embed_dim)
        # context shape: (batch_size, seq_len, context_dim)

        ##### 1. LayerNorm + SelfAtten + ResConnect
        # (batch_size, seq_len, embed_dim)
        res = x
        x = self.norm1(x)
        x = self.attn1(x)
        x += res

        ##### 2. LayerNorm + CrossAtten + ResConnect
        # (batch_size, seq_len, embed_dim)
        res = x
        x = self.norm2(x)
        x = self.attn2(x, context)
        x += res

        ##### 3. LayerNorm + FeedForward + ResConnect
        # (batch_size, seq_len, embed_dim)
        res = x
        x = self.norm3(x)
        x = self.ff(x)
        x += res
        
        return x

class UNetAttentionBlock(nn.Module):
    """UNetAttention 块：包含多个 UNetTransformerBlock 层的堆叠，以及输入和输出的额外处理。"""

    def __init__(self, n_head: int, embed_dim: int, context_dim: int = 768) -> None:
        super().__init__()
        
        in_channels = embed_dim * n_head

        # Pre-Process
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [UNetTransformerBlock(n_head, in_channels, context_dim)]
        )

        # Output process
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x shape:       (batch_size, in_channels, H, W)
        # context shape: (batch_size, seq_len, context_dim)
        b, c, h, w = x.shape
        res_long = x

        ##### 1. 处理输入特征图
        # (batch_size, in_channels, H, W)
        x = self.norm(x)
        x = self.proj_in(x)

        ##### 2. 合并像素维度以应用注意力
        # (batch_size, H*W, in_channels)
        x = x.view(b, c, h * w).transpose(-1, -2)

        ##### 3. 传入 Transformer blocks: SelfAtten + CrossAtten + FeedForward
        # (batch_size, H*W, in_channels)
        for block in self.transformer_blocks:
            x = block(x, context)

        ##### 4. 还原为输入形状
        # (batch_size, in_channels, H, W)
        x = x.transpose(-1, -2).view(b, c, h, w)

        ##### 5. 处理输出特征图
        # (batch_size, in_channels, H, W)
        x = self.proj_out(x)

        ##### 6. 残差连接
        # (batch_size, in_channels, H, W)
        x += res_long
        
        return x

class UNetDownSample(nn.Module):
    """UNet 下采样层：通过大 3×3 卷积核、步长为 2 的卷积将特征图空间尺寸减半。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, height, width) -> (batch_size, channels, height // 2, width // 2)
        return self.op(x)

class UNetUpSample(nn.Module):
    """UNet 上采样层：通过邻近插值将空间尺寸加倍，再通过一个卷积层。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, height, width) -> (batch_size, channels, height * 2, width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x

class SwitchSequential(nn.Sequential):
    """用于在单次前向传递中集成不同的层类型，在传递所有输入时根据类型自动路由额外的参数。"""

    def forward(self, x: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        return x

class UNetOutputLayer(nn.Sequential):
    """UNet 输出层：将最终特征图映射回原始通道。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        # (batch_size, 320, H, W) -> (batch_size, 4, H, W)
        super().__init__(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

class DiffusionModel(nn.Module):
    """扩散模型：输入 VAE 编码器产生的潜空间特征图、CLIP 模型产生的提示词序列嵌入、指定时间步 t 的正余弦编码，通过 UNet 模型将三者信息融合，最终输出 t-1 时间步的预测噪声。"""

    def __init__(self) -> None:
        super().__init__()
        
        self.time_embed = TimeEmbedding(320)

        self.input_blocks = nn.ModuleList(
            [
                ##### 1. (batch_size, 4, height // 8, width // 8) -> (batch_size, 320, height // 8, width // 8)
                SwitchSequential(
                    nn.Conv2d(4, 320, kernel_size=3, padding=1)
                ),
                ##### 2. (batch_size, 320, height // 8, width // 8) -> (batch_size, 320, height // 8, width // 8)
                SwitchSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                ##### 3. (batch_size, 320, height // 8, width // 8) -> (batch_size, 320, height // 8, width // 8)
                SwitchSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                ##### 4. (batch_size, 320, height // 8, width // 8) -> (batch_size, 320, height // 16, width // 16)
                SwitchSequential(
                    UNetDownSample(320)
                ),
                ##### 5. (batch_size, 320, height // 16, width // 16) -> (batch_size, 640, height // 16, width // 16)
                SwitchSequential(
                    UNetResidualBlock(320, 640), UNetAttentionBlock(8, 80)
                ),
                ##### 6. (batch_size, 640, height // 16, width // 16) -> (batch_size, 640, height // 16, width // 16)
                SwitchSequential(
                    UNetResidualBlock(640, 640), UNetAttentionBlock(8, 80)
                ),
                ##### 7. (batch_size, 640, height // 16, width // 16) -> (batch_size, 640, height // 32, width // 32)
                SwitchSequential(
                    UNetDownSample(640)
                ),
                ##### 8. (batch_size, 640, height // 32, width // 32) -> (batch_size, 1280, height // 32, width // 32)
                SwitchSequential(
                    UNetResidualBlock(640, 1280), UNetAttentionBlock(8, 160)
                ),
                ##### 9. (batch_size, 1280, height // 32, width // 32) -> (batch_size, 1280, height // 32, width // 32)
                SwitchSequential(
                    UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)
                ),
                ##### 10. (batch_size, 1280, height // 32, width // 32) -> (batch_size, 1280, height // 64, width // 64)
                SwitchSequential(
                    UNetDownSample(1280)
                ),
                ##### 11. (batch_size, 1280, height // 64, width // 64) -> (batch_size, 1280, height // 64, width // 64)
                SwitchSequential(
                    UNetResidualBlock(1280, 1280)
                ),
                ##### 12. (batch_size, 1280, height // 64, width // 64) -> (batch_size, 1280, height // 64, width // 64)
                SwitchSequential(
                    UNetResidualBlock(1280, 1280)
                ),
            ]
        )

        # (batch_size, 1280, height // 64, width // 64)
        self.middle_block = SwitchSequential(
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280),
        )

        self.output_blocks = nn.ModuleList(
            [
                ##### 1. (batch_size, 1280+1280, height // 64, width // 64) -> (batch_size, 1280, height // 64, width // 64)
                SwitchSequential(
                    UNetResidualBlock(2560, 1280)
                ),
                ##### 2. (batch_size, 1280+1280, height // 64, width // 64) -> (batch_size, 1280, height // 64, width // 64)
                SwitchSequential(
                    UNetResidualBlock(2560, 1280)
                ),
                ##### 3. (batch_size, 1280+1280, height // 64, width // 64) -> (batch_size, 1280, height // 32, width // 32)
                SwitchSequential(
                    UNetResidualBlock(2560, 1280), 
                    UNetUpSample(1280)
                ),
                ##### 4. (batch_size, 1280+1280, height // 32, width // 32) -> (batch_size, 1280, height // 32, width // 32)
                SwitchSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                ##### 5. (batch_size, 1280+1280, height // 32, width // 32) -> (batch_size, 1280, height // 32, width // 32)
                SwitchSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                ##### 6. (batch_size, 1280+640, height // 32, width // 32) -> (batch_size, 1280, height // 16, width // 16)
                SwitchSequential(
                    UNetResidualBlock(1920, 1280), UNetAttentionBlock(8, 160), 
                    UNetUpSample(1280),
                ),
                ##### 7. (batch_size, 1280+640, height // 16, width // 16) -> (batch_size, 640, height // 16, width // 16)
                SwitchSequential(
                    UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)
                ),
                ##### 8. (batch_size, 640+640, height // 16, width // 16) -> (batch_size, 640, height // 16, width // 16)
                SwitchSequential(
                    UNetResidualBlock(1280, 640), UNetAttentionBlock(8, 80)
                ),
                ##### 9. (batch_size, 640+320, height // 16, width // 16) -> (batch_size, 640, height // 8, width // 8)
                SwitchSequential(
                    UNetResidualBlock(960, 640), UNetAttentionBlock(8, 80), 
                    UNetUpSample(640),
                ),
                ##### 10. (batch_size, 640+320, height // 8, width // 8) -> (batch_size, 320, height // 8, width // 8)
                SwitchSequential(
                    UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)
                ),
                ##### 11. (batch_size, 320+320, height // 8, width // 8) -> (batch_size, 320, height // 8, width // 8)
                SwitchSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)
                ),
                ##### 12. (batch_size, 320+320, height // 8, width // 8) -> (batch_size, 320, height // 8, width // 8)
                SwitchSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)
                ),
            ]
        )

        self.out = UNetOutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # latent shape:  (batch_size, 4, height // 8, width // 8)
        # context shape: (batch_size, 77, 768)
        # t shape:       (1, 320)
        skip_connections: list[torch.Tensor] = []

        ##### 1. 时间嵌入
        # (1, 320) -> (1, 1280)
        t = self.time_embed(t)
        
        ##### 2. Encoder
        # (batch_size, 4, height // 8, width // 8) -> (batch_size, 1280, height // 64, width // 64)
        for layer in self.input_blocks:
            latent = layer(latent, context, t)
            skip_connections.append(latent)

        ##### 3. Bottleneck
        # (batch_size, 1280, height // 64, width // 64)
        latent = self.middle_block(latent, context, t)

        ##### 4. Decoder
        # (batch_size, 1280, height // 64, width // 64) -> (batch_size, 320, height // 8, width // 8)
        for layer in self.output_blocks:
            latent = torch.cat((latent, skip_connections.pop()), dim=1)
            latent = layer(latent, context, t)

        ##### 5. 输出映射
        # (batch_size, 320, height // 8, width // 8) -> (batch_size, 4, height // 8, width // 8)
        output = self.out(latent)

        return output
