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


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    # def build_mask(self, **kw):
    #     assert 'context' in kw and 'key_len' in kw and 'bsz' in kw
    #     context, key_len, bsz = kw['context'], kw['key_len'], kw['bsz']
    #
    #     if context.size(0) != bsz:
    #         assert context.size(1) == bsz
    #         context = context.transpose(0, 1)
    #
    #     """ position based -- gamma"""
    #     forward_x = self.forward_position_x
    #     backward_x = self.backward_position_x
    #     self_x = torch.zeros(1).type_as(backward_x)
    #     position_x = torch.cat([forward_x, self_x, backward_x], 0)
    #
    #     max_x_len = position_x.size(-1)
    #     half_max_x_len = (max_x_len + 1) // 2
    #
    #     indices = torch.arange(key_len).unsqueeze(0).repeat(key_len, 1). \
    #         add(torch.arange(half_max_x_len - key_len, half_max_x_len).flip([0]).unsqueeze(1)). \
    #         view(-1).long().cuda()
    #     position_x = position_x.view(-1)[indices].view(key_len, key_len)
    #
    #     """ head based -- gamma"""
    #     head_x = self.head_x
    #
    #     """position and head based -- gamma
    #       gamma num_heads * key_len * key_len"""
    #     x = position_x.unsqueeze(0).repeat(self.num_heads, 1, 1). \
    #         add(head_x.unsqueeze(-1).unsqueeze(-1))
    #
    #     """ context weight based -- x"""
    #     context_x = self.context_x_fc(context).unsqueeze(1)
    #
    #     log_weights = context_x.add(x.unsqueeze(0).repeat(bsz, 1, 1, 1)).sigmoid().clamp(1e-10).log()
    #
    #     if key_len == context.size(1):
    #         return log_weights
    #     else:
    #         return log_weights[:, :, -1].unsqueeze(2)
    #
    # def reset_parameters(self):
    #     if self.forward_position_x is not None:
    #         nn.init.normal_(self.forward_position_x)
    #     if self.backward_position_x is not None:
    #         nn.init.normal_(self.backward_position_x)
    #     if self.head_x is not None:
    #         nn.init.normal_(self.head_x)
    #     if self.context_x_fc is not None:
    #         nn.init.xavier_uniform_(self.context_x_fc.weight)

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

    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)
        self.mask_selection = mask_selection
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio
        # torch.set_printoptions(precision=None, threshold=5000, edgeitems=None, linewidth=300, profile=None,
        #                        sci_mode=None)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        # print(f'Using {self.mask_selection}!!! !!!')
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if self.use_DropKey:
            m_r = torch.ones_like(score) * self.mask_ratio
            score = score + torch.bernoulli(m_r) * -1e12

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        ## DropKey for Vision Transformer

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
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
                 upper_offset: int = 20, lower_offset: int = -20, atten_sel: str = 'conformer', use_DropKey=False,
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
                                                        dman_max_len=dman_max_len,
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
        if self.mask_selection != 'dman':
            outputs = self.attention(inputs, inputs, inputs, pos_embedding=self.pos_embedding, mask=self.mask)

        return self.dropout(outputs)
