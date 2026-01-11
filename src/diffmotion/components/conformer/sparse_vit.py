
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


# ---------------------------
# 🔹 引入 Gumbel Top-K Sparse Gate
# ---------------------------
class GumbelTopKGate(nn.Module):
    def __init__(self, d_model, K=16, temp=1.0):
        super().__init__()
        self.K = K
        self.temp = temp
        # 用于学习补丁重要性的投影层
        self.proj = nn.Linear(d_model, 1)

    def forward(self, q, k):
        """
        q, k : [B, heads, N, d_head]
        返回一个可微的二进制掩码 [B, 1, N, N]
        """
        # 计算原始注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # [B, heads, N, N]

        # 跨头平均以获得统一的稀疏模式
        logits = attn_scores.mean(dim=1)  # [B, N, N]

        # Gumbel Softmax 技巧实现可微 Top-K 选择
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        probs = F.softmax((logits + gumbel) / self.temp, dim=-1)

        # 动态选择 Top-K 个补丁
        # 如果序列长度小于 K，则不进行稀疏化（或调整 K）
        topk = torch.topk(probs, self.K, dim=-1)[0]
        thresh = topk.min(dim=-1, keepdim=True)[0]
        sparse_mask = (probs >= thresh).float()  # [B, N, N]
        return sparse_mask.unsqueeze(1)  # 扩展为 [B, 1, N, N] 以进行广播


# ---------------------------
# 🔹 修改后的 Sparse Relative Multi-Head Attention
# ---------------------------
class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
            mask_selection: str = 'causalMask',
            position_embedding_type: str = 'Transformer_FX',
            use_DropKey: bool = False,
            mask_ratio: float = 0.3,
            K: int = 16,  # 新增：稀疏性参数 K
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = d_model
        self.K = K

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

        # Trainable attention head importance scores
        self.head_scores = nn.Parameter(torch.ones(num_heads))
        self.gate = GumbelTopKGate(d_model // num_heads, K)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:

        B, N, C = value.shape   # 【B,N,C】 = [18,50,1280]

        # Standard QKV projection
        qkv = self.qkv(value).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]    # 【B,heads,N,dim】 = [18,16,50,80]

        # Get sparse attention mask via gating
        attn_mask = self.gate(q, k)  # [B, 1, N, N]

        # Compute masked attention
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1) * attn_mask
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, C)

        # Weighted by learned head importance
        return self.proj(out * self.head_scores.mean())
        # 投影与重塑形
        # batch_size = value.size(0)
        # [B, N, heads, d_head] -> permute -> [B, heads, N, d_head]
        q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        p = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        # 🔹 生成稀疏掩码
        # 使用 q 和 k 计算哪些 key 是重要的
        sparse_mask = self.gate(q, k)  # [B, 1, N, N]

        # 计算内容分数 (Content Score)
        # (q + u) * k^T
        content_score = torch.matmul((q + self.u_bias.unsqueeze(1)), k.transpose(2, 3))

        # 计算位置分数 (Position Score)
        # (q + v) * p^T
        pos_score = torch.matmul((q + self.v_bias.unsqueeze(1)), p.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        # 合并分数并缩放
        score = (content_score + pos_score) / self.sqrt_dim

        # 🔹 应用稀疏掩码
        # 将非 Top-K 的位置填充为极小值
        score = score.masked_fill(sparse_mask == 0, -1e9)

        # 应用外部 Mask (如 causalMask)
        # if mask is not None:
        #     if mask.dim() == 3:
        #         mask = mask.unsqueeze(1)
        #     score.masked_fill_(mask, -1e9)

        # DropKey 逻辑
        # if self.use_DropKey:
        #     m_r = torch.ones_like(score) * self.mask_ratio
        #     score = score + torch.bernoulli(m_r) * -1e12

        # Softmax 与输出
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score


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
                 dman_max_len=200,
                 position_embedding_type: str = 'Transformer_FX', causal_mask_diagonal: int = 0,
                 upper_offset: int = 20, lower_offset: int = -20, atten_sel: str = 'conformer',
                 use_DropKey=False,
                    mask_ratio=0.3,):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.pos_embedding = None
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.atten_sel = atten_sel
        self.num_heads = num_heads
        if self.atten_sel == "conformer":
            self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p,
                                                        mask_selection=mask_selection,
                                                        position_embedding_type=position_embedding_type,
                                                        use_DropKey=use_DropKey,
                                                        mask_ratio=mask_ratio,
                                                        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.mask_selection = mask_selection
        self.position_embedding = position_embedding_type
        self.mask = None
        self.causal_mask_diagonal = causal_mask_diagonal
        self.upper_offset = upper_offset
        self.lower_offset = lower_offset
        # print(f'atten_sel: {self.atten_sel} + inf_pos: {self.inf_pos} --------------------------------')
        # if self.mask_selection == 'dman' and self.atten_sel != 'informer':
        #     self.mask = mask_strategy.DMAN(max_len=dman_max_len, embed_dim=d_model, num_heads=num_heads)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()

        self.pos_embedding = self.positional_encoding(seq_length)
        self.pos_embedding = self.pos_embedding.repeat(batch_size, 1, 1)
        if self.mask is None:
            if self.mask_selection == 'causalMask':
                self.mask = mask_strategy.causal_mask(batch_size, seq_length, self.causal_mask_diagonal)
            elif self.mask_selection == 'diagonal_matrix':
                print(f'upper_offset: {self.upper_offset}, lower_offset: {self.lower_offset}......')
                self.mask = mask_strategy.upper_lower_diagonal_mask(batch_size, seq_length, self.upper_offset,
                                                                    self.lower_offset)
            elif self.mask_selection == 'no_mask':
                self.mask = None
        if self.mask is not None and self.mask_selection != 'dman':
            if self.mask.shape[0] != batch_size:
                self.mask = self.mask[0].unsqueeze(0).repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=self.pos_embedding, mask=self.mask)

        return self.dropout(outputs)
