import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RelativeMultiHeadAttention(nn.Module):
    """
    Optimized Relative Multi-Head Attention (Transformer-XL style).
    Reduces memory footprint by using broadcasting instead of repeating tensors.
    """

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
            use_DropKey: bool = False,
            mask_ratio: float = 0.3,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(d_model)  # Pre-compute scale

        # Linear Projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.out_proj = nn.Linear(d_model, d_model)

        # Global Biases (Parameters)
        # Shape: (num_heads, d_head) - We will broadcast this during forward
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))

        # Initialization
        self._reset_parameters()

        # Regularization
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.pos_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

    def _relative_shift(self, pos_score: torch.Tensor) -> torch.Tensor:
        """
        Optimized relative shift implementation using padding.
        Input: (B, H, L, L)
        Output: (B, H, L, L)
        """
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()

        # Pad one column of zeros to the right: (B, H, L, L+1)
        # This is more memory efficient than creating a new tensor
        padded = F.pad(pos_score, (1, 0))

        # Reshape to (B, H, L+1, L) and slice
        padded = padded.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        return padded[:, :, 1:].view_as(pos_score)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            pos_embedding: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: (Batch, Seq_Len, Dim)
            pos_embedding: (1, Seq_Len, Dim) or (Seq_Len, Dim) - NOT Repeated
            mask: (Batch, 1, L, L) or (1, 1, L, L)
        """
        batch_size = value.size(0)

        # 1. Projections
        # Result shapes: (Batch, Heads, Seq, Dim_Head) -> (B, H, L, D)
        # Note: We permute to (B, H, L, D) for easier broadcasting with biases
        q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Pos embedding projection (Shared across batch)
        # Input assumed (L, Dim) or (1, L, Dim) -> Target: (1, H, L, D)
        if pos_embedding.dim() == 2:
            pos_embedding = pos_embedding.unsqueeze(0)
        p = self.pos_proj(pos_embedding).view(1, -1, self.num_heads, self.d_head).transpose(1, 2)

        # 2. Content Score: AC = (Q + u) * K^T
        # Optimization: Decompose into (Q * K^T) + (u * K^T) to avoid broadcasting Q+u (huge memory)

        # (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
        content_score = torch.matmul(q, k.transpose(-2, -1))

        # (1, H, 1, D) @ (B, H, D, L) -> (B, H, 1, L) -- Broadcast add
        u_bias_term = torch.matmul(
            self.u_bias.view(1, self.num_heads, 1, self.d_head),
            k.transpose(-2, -1)
        )
        content_score = content_score + u_bias_term

        # 3. Position Score: BD = (Q + v) * P^T
        # Optimization: Decompose into (Q * P^T) + (v * P^T)

        # (B, H, L, D) @ (1, H, D, L) -> (B, H, L, L)
        pos_score = torch.matmul(q, p.transpose(-2, -1))

        # (1, H, 1, D) @ (1, H, D, L) -> (1, H, 1, L) -- Broadcast add
        v_bias_term = torch.matmul(
            self.v_bias.view(1, self.num_heads, 1, self.d_head),
            p.transpose(-2, -1)
        )
        pos_score = pos_score + v_bias_term

        # Relative Shift
        pos_score = self._relative_shift(pos_score)

        # 4. Combine
        # In-place add
        score = content_score
        score.add_(pos_score).mul_(self.scale)

        # 5. Masking
        if mask is not None:
            # Ensure mask matches score dimensions for broadcasting
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, L, L)
            # Use -inf for numerical stability (works with FP16)
            score.masked_fill_(mask, -torch.inf)

        # 6. DropKey (Only during training!)
        if self.use_DropKey and self.training:
            # Generate mask only when needed
            drop_mask = torch.bernoulli(torch.full_like(score, self.mask_ratio)).to(torch.bool)
            score.masked_fill_(drop_mask, -torch.inf)

        # 7. Attention Probability
        attn = F.softmax(score, dim=-1)

        # Dropout (Only during training)
        if self.training:
            attn = self.dropout(attn)

        # 8. Context
        # (B, H, L, L) @ (B, H, L, D) -> (B, H, L, D)
        context = torch.matmul(attn, v)

        # Combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, MaxLen, Dim)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        return self.pe[:, :seq_len, :]


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Wrapper module that handles LayerNorm, Positional Encoding generation, and Masking.
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout_p: float = 0.1,
            mask_selection: str = 'causalMask',
            use_DropKey: bool = False,
            mask_ratio: float = 0.3,
            **kwargs  # Catch unused args from legacy code
    ):
        super(MultiHeadedSelfAttentionModule, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.attention = RelativeMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_p=dropout_p,
            use_DropKey=use_DropKey,
            mask_ratio=mask_ratio
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.mask_selection = mask_selection

    def _get_mask(self, batch_size: int, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """Simple mask generator helper"""
        if self.mask_selection == 'causalMask':
            # Create a Lower Triangular Mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L) - Broadcastable
        return None

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        inputs: (Batch, Seq_Len, Dim)
        mask: Optional external mask
        """
        batch_size, seq_length, _ = inputs.size()

        # 1. Get Positional Embedding
        # Shape: (1, Seq_Len, Dim) -- No Repeat!
        pos_embedding = self.positional_encoding(seq_length)

        # 2. Handle Mask
        if mask is None:
            mask = self._get_mask(batch_size, seq_length, inputs.device)

        # 3. Layer Norm (Pre-Norm style is standard for Conformer/modern Transformers)
        x = self.layer_norm(inputs)

        # 4. Attention
        outputs = self.attention(
            query=x,
            key=x,
            value=x,
            pos_embedding=pos_embedding,
            mask=mask
        )

        # 5. Output Dropout
        return self.dropout(outputs)


# --- 简单的测试代码 (Usage Example) ---
if __name__ == "__main__":
    # 配置
    B, L, D, H = 2, 100, 256, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型
    model = MultiHeadedSelfAttentionModule(
        d_model=D,
        num_heads=H,
        dropout_p=0.1,
        mask_selection='causalMask',
        use_DropKey=True  # 开启 DropKey 测试训练模式
    ).to(device)

    # 伪造输入
    x = torch.randn(B, L, D).to(device)

    # 1. 训练模式测试
    model.train()
    out_train = model(x)
    print(f"Training Output Shape: {out_train.shape}")  # Should be (2, 100, 256)

    # 2. 推理模式测试 (速度应该更快，不执行 DropKey)
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print(f"Inference Output Shape: {out_eval.shape}")

    print("\nDone. Memory optimized version.")