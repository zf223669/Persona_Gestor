
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import numpy as np

from .embedding import PositionalEncoding
from .modules import Linear
import src.diffmotion.components.mask.mask as mask_strategy
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math


class FastRelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # 整合 QKV 投影以提高效率
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout_p

    def forward(
            self,
            x: Tensor,  # 假设输入是 [B, N, C]
            mask: Optional[Tensor] = None,  # [B, N, N] 或 [B, 1, N, N]
    ) -> Tensor:
        B, N, C = x.shape

        # 1. 投影并重塑: [B, N, 3*C] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 使用 PyTorch 2.0 FlashAttention 内核
        # SDPA 会自动处理 scaling (1/sqrt(dk)) 和 dropout
        # 注意：如果 mask 是布尔类型，FlashAttention 支持效果最好
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True  # 如果是自回归模型请设为 True
            )

        # 3. 合并头并投影回输出
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(out)


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, mask_selection: str = 'causalMask',
                 position_embedding_type: str = 'Transformer_FX', causal_mask_diagonal: int = 0,
                 upper_offset: int = 20, lower_offset: int = -20, atten_sel: str = 'conformer',):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.pos_embedding = None
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.atten_sel = atten_sel
        self.num_heads = num_heads
        self.attention = FastRelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        self.mask_selection = mask_selection
        self.position_embedding = position_embedding_type
        self.mask = None
        self.causal_mask_diagonal = causal_mask_diagonal
        self.upper_offset = upper_offset
        self.lower_offset = lower_offset

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()

        # self.pos_embedding = self.positional_encoding(seq_length)
        # self.pos_embedding = self.pos_embedding.repeat(batch_size, 1, 1)
        if self.mask is None:
            if self.mask_selection == 'causalMask':
                self.mask = mask_strategy.causal_mask(batch_size, seq_length, self.causal_mask_diagonal)
            elif self.mask_selection == 'diagonal_matrix':
                print(f'upper_offset: {self.upper_offset}, lower_offset: {self.lower_offset}......')
                self.mask = mask_strategy.upper_lower_diagonal_mask(batch_size, seq_length, self.upper_offset,
                                                                    self.lower_offset)
            elif self.mask_selection == 'no_mask':
                self.mask = None
        if self.mask is not None:
            if self.mask.shape[0] != batch_size:
                self.mask = self.mask[0].unsqueeze(0).repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, mask=mask)

        return self.dropout(outputs)
