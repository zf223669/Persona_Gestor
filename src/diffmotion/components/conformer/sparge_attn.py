# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .embedding import PositionalEncoding
from .modules import Linear
import src.diffmotion.components.mask.mask as mask_strategy
from torch.nn.parameter import Parameter

try:
    from src.diffmotion.components.spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
except ImportError:
    print("请先安装 SpargeAttn: pip install ninja && python setup.py install")


class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
            mask_selection: str = 'causalMask',
            position_embedding_type: str = 'Transformer_FX',
            dman_max_len: int = 400,
            use_DropKey: bool = False,
            mask_ratio: float = 0.3,
            # 新增 topk 参数，用于平衡精度与稀疏度（0.1~1.0）[3, 6]
            sparge_topk: float = 0.5
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)  # 注意：缩放因子通常基于 d_head

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)
        self.sparge_topk = sparge_topk
        self.is_causal = (mask_selection == 'causalMask')

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        # 投影变换并重塑形状为 (batch, heads, seq_len, head_dim) [4]
        q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Transformer-XL 特有的相对位置偏置注入 [7]
        # 注意：SpargeAttn 核心加速在于跳过 QK^T 的块计算。
        # 如果需要保留完全一致的 Transformer-XL 行为，建议使用 block_sparse_sage2_attn_cuda
        # 这里展示如何使用推荐的即插即用 API 来获得 2.5x-5x 的加速 [8]

        # 应用 SpargeAttn [3, 9]
        # 它会自动执行两阶段在线过滤（标记压缩预测 + 在线 Softmax 过滤）[10, 11]
        context = spas_sage2_attn_meansim_topk_cuda(
            q, k, v,
            topk=self.sparge_topk,
            tensor_layout='HND',
            is_causal=self.is_causal
        )

        # 恢复形状并输出
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
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
                 informer_factor=5, use_DropKey=False,
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
