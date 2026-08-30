import torch
from torch import nn
import torch.nn.functional as F

class CLIPEmbedding(nn.Module):
    """
    CLIP 词嵌入层：对输入序列编号完成词嵌入和位置编码。
    Args:
        vocab_size: 49408
        embed_dim: 768
        n_tokens: 77
    """

    def __init__(self, vocab_size: int, embed_dim: int, n_tokens: int) -> None:
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(n_tokens, embed_dim)
        
        # (1, n_tokens)
        self.register_buffer("position_ids", torch.arange(n_tokens).unsqueeze(0), persistent=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        ##### 1. 词嵌入
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.token_embedding(input_ids)

        ##### 2. 位置编码
        # (batch_size, seq_len, embed_dim) += (1, n_tokens, embed_dim)
        x += self.position_embedding(self.position_ids)

        return x

class CLIPAttention(nn.Module):
    """
    CLIP 注意力层：计算输入序列对自身的注意力修正。
    Args:
        num_heads: 12
        embed_dim: 768
    """

    def __init__(self, num_heads: int, embed_dim: int) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:        
        # (batch_size, seq_len, embed_dim)
        b, _, _ = input_shape = x.shape

        ##### 1. 投影到 q, k, v 空间
        # (batch_size, seq_len, embed_dim)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        ##### 2. 分割为各个头，同时前置头维度
        # q,k,v shape:  (batch_size, num_heads, seq_len, head_dim)
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        ##### 3. 使用 PyTorch 内置函数完成注意力修正
        # output shape: (batch_size, num_heads, seq_len, head_dim)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal_mask)

        ##### 4. 还原为输入形状
        # output shape: (batch_size, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(input_shape)

        ##### 5. 输出层投影
        # output shape: (batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        return output

class CLIPMLP(nn.Module):
    """
    CLIP 前馈层：包括两层线性层及一个 QuickGELU 激活函数。
    Args:
        embed_dim: 768
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim * 4)
        x = self.fc1(x)
        # QuickGELU
        x = x * torch.sigmoid(1.702 * x)
        # (batch_size, seq_len, embed_dim * 4) -> (batch_size, seq_len, embed_dim)
        x = self.fc2(x)

        return x

class CLIPLayer(nn.Module):
    """
    CLIPLayer 层：输入序列先通过自注意力机制修正，然后送入前馈层。
    Args:
        n_heads: 12
        embed_dim: 768
    """

    def __init__(self, n_heads: int, embed_dim: int) -> None:
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = CLIPAttention(n_heads, embed_dim)
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = CLIPMLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ##### 1. LayerNorm + SelfAtten + ResConnect
        # (batch_size, seq_len, embed_dim)
        res = x
        x = self.layer_norm1(x)
        x = self.self_attn(x, causal_mask=True)
        x += res

        ##### 2. LayerNorm + FeedForward + ResConnect
        # (batch_size, seq_len, embed_dim)
        res = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x += res

        return x

class CLIPEncoder(nn.Module):
    """
    CLIP 编码器层：由 12 层 CLIPLayer 层堆叠而成。
    Args:
        n_layers: 12
        n_heads: 12
        embed_dim: 768
    """

    def __init__(self, n_layers: int, n_heads: int, embed_dim: int) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([CLIPLayer(n_heads, embed_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, embed_dim)
        for layer in self.layers:
            x = layer(x)

        return x

class CLIP(nn.Module):
    """CLIP 模型：输入序列编号先进行词嵌入，再通过多层自注意力机制和前馈网络，最终生成嵌入表示。"""

    def __init__(self) -> None:
        super().__init__()
        
        self.embeddings = CLIPEmbedding(49408, 768, 77)
        self.encoder = CLIPEncoder(12, 12, 768)
        self.final_layer_norm = nn.LayerNorm(768)

    def forward(self, input_ids: torch.LongTensor | torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len)
        input_ids = input_ids.to(torch.long)

        ##### 1. 词嵌入
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        hidden_state = self.embeddings(input_ids)

        ##### 2. CLIP layers
        # (batch_size, seq_len, embed_dim)
        hidden_state = self.encoder(hidden_state)

        ##### 3. 层归一化
        # (batch_size, seq_len, embed_dim)
        hidden_state = self.final_layer_norm(hidden_state)

        return hidden_state