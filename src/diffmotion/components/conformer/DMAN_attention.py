# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from fairseq import utils

# from fairseq import utils


class DMAN_attention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, re_weight_m=0, max_len=1024,
                 activation=torch.nn.ReLU, activation_dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        """ re-weight the score of attention matrix"""
        assert isinstance(re_weight_m, int) and re_weight_m >= 0
        self.re_weight_m = re_weight_m

        if self.re_weight_m == 1:
            self.forward_position_x = Parameter(torch.Tensor(max_len))
            self.backward_position_x = Parameter(torch.Tensor(max_len))
            self.head_x = Parameter(torch.Tensor(self.num_heads))
            self.context_x_fc = nn.Linear(self.embed_dim, 1, False)
            self.indices = torch.arange(max_len, dtype=torch.int32).unsqueeze(0). \
                add(torch.arange(0, max_len + 1, dtype=torch.int32).flip([0]).unsqueeze(1))
        else:
            self.forward_position_x = self.backward_position_x = self.head_x = self.context_x_fc = None
            self.indices = None
        self.activation = nn.ReLU()
        self.activation_dropout = activation_dropout

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.kaiming_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.forward_position_x is not None:
            nn.init.normal_(self.forward_position_x)
        if self.backward_position_x is not None:
            nn.init.normal_(self.backward_position_x)
        if self.head_x is not None:
            nn.init.normal_(self.head_x)
        if self.context_x_fc is not None:
            nn.init.xavier_uniform_(self.context_x_fc.weight)

    def build_mask(self, **kw):

        if self.re_weight_m == 1:
            assert 'context' in kw and 'key_len' in kw and 'bsz' in kw
            context, key_len, bsz = kw['context'], kw['key_len'], kw['bsz']

            if context.size(0) != bsz:
                assert context.size(1) == bsz
                context = context.transpose(0, 1)

            """ position based -- gamma"""
            forward_x = self.forward_position_x
            backward_x = self.backward_position_x
            self_x = torch.zeros(1).type_as(backward_x)
            position_x = torch.cat([forward_x, self_x, backward_x], 0)

            indices = self.indices[:key_len, :key_len].cuda()

            position_x = torch.take(position_x, indices.long()).view(key_len, key_len)

            """ head based -- gamma"""
            head_x = self.head_x

            """position and head based -- gamma
              gamma num_heads * key_len * key_len"""
            x = position_x.unsqueeze(0).repeat(self.num_heads, 1, 1). \
                add(head_x.unsqueeze(-1).unsqueeze(-1))

            """ context weight based -- x"""
            context_x = self.context_x_fc(context).unsqueeze(1)

            log_weights = context_x.add(x.unsqueeze(0).repeat(bsz, 1, 1, 1)).sigmoid()
            log_weights = log_weights.div(log_weights.max(-1, keepdim=True)[0].detach()).clamp(1e-10).log()

            if key_len == context.size(1):
                return log_weights
            else:
                return log_weights[:, :, -1].unsqueeze(2)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        # q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None or self.re_weight_m == 1:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.re_weight_m == 1:
                log_weights = self.build_mask(context=query, key_len=src_len, bsz=bsz)  # Importance Function
                attn_weights += log_weights
            if key_padding_mask is not None:
                if self.onnx_trace:
                    attn_weights = torch.where(
                        key_padding_mask.unsqueeze(1).unsqueeze(2),
                        torch.Tensor([float("-Inf")]),
                        attn_weights.float()
                    ).type_as(attn_weights)
                else:
                    attn_weights = attn_weights.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(2),
                        float('-inf'),
                    )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.activation(attn)
        attn = F.dropout(attn, p=self.activation_dropout, training=self.training)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        pass
        # return utils.get_incremental_state(
        #     self,
        #     incremental_state,
        #     'attn_state',
        # ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        pass
        # utils.set_incremental_state(
        #     self,
        #     incremental_state,
        #     'attn_state',
        #     buffer,
        # )

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights

    #########################################################################################33
    # """Multi-headed attention.
    #
    # See "Attention Is All You Need" for more details.
    # """
    #
    # def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
    #              add_zero_attn=False, re_weight_m=0, max_len=1024):
    #     super().__init__()
    #     self.embed_dim = embed_dim
    #     self.num_heads = num_heads
    #     self.dropout = dropout
    #     self.head_dim = embed_dim // num_heads
    #     assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
    #     self.scaling = self.head_dim ** -0.5
    #
    #     self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
    #     if bias:
    #         self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
    #     else:
    #         self.register_parameter('in_proj_bias', None)
    #     self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    #
    #     if add_bias_kv:
    #         self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
    #         self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
    #     else:
    #         self.bias_k = self.bias_v = None
    #
    #     self.add_zero_attn = add_zero_attn
    #
    #     """ re-weight the score of attention matrix"""
    #     assert isinstance(re_weight_m, int) and re_weight_m >= 0
    #     self.re_weight_m = re_weight_m
    #
    #     if self.re_weight_m == 1:
    #         self.forward_position_x = Parameter(torch.Tensor(max_len))
    #         self.backward_position_x = Parameter(torch.Tensor(max_len))
    #         self.head_x = Parameter(torch.Tensor(self.num_heads))
    #         self.context_x_fc = nn.Linear(self.embed_dim, 1, False)
    #     else:
    #         self.forward_position_x = self.backward_position_x = self.head_x = self.context_x_fc = None
    #
    #     self.reset_parameters()
    #
    #     self.onnx_trace = False
    #
    # def prepare_for_onnx_export_(self):
    #     self.onnx_trace = True
    #
    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.in_proj_weight)
    #     nn.init.kaiming_uniform_(self.out_proj.weight)
    #     if self.in_proj_bias is not None:
    #         nn.init.constant_(self.in_proj_bias, 0.)
    #         nn.init.constant_(self.out_proj.bias, 0.)
    #     if self.bias_k is not None:
    #         nn.init.xavier_normal_(self.bias_k)
    #     if self.bias_v is not None:
    #         nn.init.xavier_normal_(self.bias_v)
    #     if self.forward_position_x is not None:
    #         nn.init.normal_(self.forward_position_x)
    #     if self.backward_position_x is not None:
    #         nn.init.normal_(self.backward_position_x)
    #     if self.head_x is not None:
    #         nn.init.normal_(self.head_x)
    #     if self.context_x_fc is not None:
    #         nn.init.xavier_uniform_(self.context_x_fc.weight)
    #         # nn.init.constant_(self.context_x.bias, 0.)
    #
    # def build_mask(self, **kw):
    #
    #     if self.re_weight_m == 1:
    #
    #         assert 'context' in kw and 'key_len' in kw and 'bsz' in kw
    #         context, key_len, bsz = kw['context'], kw['key_len'], kw['bsz']
    #
    #         if context.size(0) != bsz:
    #             assert context.size(1) == bsz
    #             context = context.transpose(0, 1)
    #
    #         """ position based -- gamma"""
    #         forward_x = self.forward_position_x
    #         backward_x = self.backward_position_x
    #         self_x = torch.zeros(1).type_as(backward_x)
    #         position_x = torch.cat([forward_x, self_x, backward_x], 0)
    #
    #         max_x_len = position_x.size(-1)
    #         half_max_x_len = (max_x_len + 1) // 2
    #
    #         indices = torch.arange(key_len).unsqueeze(0).repeat(key_len, 1). \
    #             add(torch.arange(half_max_x_len - key_len, half_max_x_len).flip([0]).unsqueeze(1)). \
    #             view(-1).long().cuda()
    #         # indices = torch.arange(key_len, device=context.device).unsqueeze(0).repeat(key_len, 1). \
    #         #     add(torch.arange(half_max_x_len - key_len, half_max_x_len).flip([0]).unsqueeze(1)). \
    #         #     view(-1).long()
    #         position_x = position_x.view(-1)[indices].view(key_len, key_len)
    #
    #         """ head based -- gamma"""
    #         head_x = self.head_x
    #
    #         """position and head based -- gamma
    #           gamma num_heads * key_len * key_len"""
    #         x = position_x.unsqueeze(0).repeat(self.num_heads, 1, 1). \
    #             add(head_x.unsqueeze(-1).unsqueeze(-1))
    #
    #         """ context weight based -- x"""
    #         context_x = self.context_x_fc(context).unsqueeze(1)
    #
    #         log_weights = context_x.add(x.unsqueeze(0).repeat(bsz, 1, 1, 1)).sigmoid().clamp(1e-10).log()
    #
    #         if key_len == context.size(1):
    #             return log_weights
    #         else:
    #             return log_weights[:, :, -1].unsqueeze(2)
    #
    # def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
    #             need_weights=True, static_kv=False, attn_mask=None):
    #     """Input shape: Time x Batch x Channel
    #
    #     Self-attention can be implemented by passing in the same arguments for
    #     query, key and value. Timesteps can be masked by supplying a T x T mask in the
    #     `attn_mask` argument. Padding elements can be excluded from
    #     the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
    #     batch x src_len, where padding elements are indicated by 1s.
    #     """
    #
    #     qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
    #     kv_same = key.data_ptr() == value.data_ptr()
    #
    #     tgt_len, bsz, embed_dim = query.size()
    #     assert embed_dim == self.embed_dim
    #     assert list(query.size()) == [tgt_len, bsz, embed_dim]
    #     assert key.size() == value.size()
    #
    #     if incremental_state is not None:
    #         saved_state = self._get_input_buffer(incremental_state)
    #         if 'prev_key' in saved_state:
    #             # previous time steps are cached - no need to recompute
    #             # key and value if they are static
    #             if static_kv:
    #                 assert kv_same and not qkv_same
    #                 key = value = None
    #     else:
    #         saved_state = None
    #
    #     if qkv_same:
    #         # self-attention
    #         q, k, v = self.in_proj_qkv(query)
    #     elif kv_same:
    #         # encoder-decoder attention
    #         q = self.in_proj_q(query)
    #         if key is None:
    #             assert value is None
    #             k = v = None
    #         else:
    #             k, v = self.in_proj_kv(key)
    #     else:
    #         q = self.in_proj_q(query)
    #         k = self.in_proj_k(key)
    #         v = self.in_proj_v(value)
    #     q = q * self.scaling
    #
    #     if saved_state is not None:
    #
    #         if 'prev_key' in saved_state:
    #             if static_kv:
    #                 k = saved_state['prev_key']
    #             else:
    #                 k = torch.cat((saved_state['prev_key'], k), dim=0)
    #         if 'prev_value' in saved_state:
    #             if static_kv:
    #                 v = saved_state['prev_value']
    #             else:
    #                 v = torch.cat((saved_state['prev_value'], v), dim=0)
    #         saved_state['prev_key'] = k
    #         saved_state['prev_value'] = v
    #
    #         self._set_input_buffer(incremental_state, saved_state)
    #
    #     if self.bias_k is not None:
    #         assert self.bias_v is not None
    #         k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
    #         v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
    #         if attn_mask is not None:
    #             attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
    #         if key_padding_mask is not None:
    #             key_padding_mask = torch.cat(
    #                 [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
    #
    #     src_len = k.size(0)
    #
    #     if key_padding_mask is not None:
    #         assert key_padding_mask.size(0) == bsz
    #         assert key_padding_mask.size(1) == src_len
    #
    #     q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
    #     k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
    #     v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
    #
    #     if self.add_zero_attn:
    #         src_len += 1
    #         k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
    #         v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
    #         if attn_mask is not None:
    #             attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
    #         if key_padding_mask is not None:
    #             key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
    #
    #     attn_weights = torch.bmm(q, k.transpose(1, 2))
    #     assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
    #
    #     if attn_mask is not None:
    #         attn_weights += attn_mask.unsqueeze(0)
    #
    #     if key_padding_mask is not None or self.re_weight_m == 1:
    #         # don't attend to padding symbols
    #         attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
    #         if self.re_weight_m == 1:
    #             log_weights = self.build_mask(context=query, key_len=src_len, bsz=bsz)
    #             attn_weights += log_weights
    #         if key_padding_mask is not None:
    #             mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
    #             attn_weights = attn_weights.float().masked_fill(
    #                 mask,
    #                 float('-inf'),
    #             ).type_as(attn_weights)  # FP16 support: cast to float and back
    #
    #         attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    #
    #     attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
    #     attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
    #
    #     attn = torch.bmm(attn_weights, v)
    #     assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
    #     attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    #     attn = attn.relu()
    #     attn = self.out_proj(attn)
    #
    #     if need_weights:
    #         # average attention weights over heads
    #         attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
    #         attn_weights = attn_weights.sum(dim=1) / self.num_heads
    #     else:
    #         attn_weights = None
    #
    #     return attn, attn_weights
    #
    # def in_proj_qkv(self, query):
    #     return self._in_proj(query).chunk(3, dim=-1)
    #
    # def in_proj_kv(self, key):
    #     return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)
    #
    # def in_proj_q(self, query):
    #     return self._in_proj(query, end=self.embed_dim)
    #
    # def in_proj_k(self, key):
    #     return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
    #
    # def in_proj_v(self, value):
    #     return self._in_proj(value, start=2 * self.embed_dim)
    #
    # def _in_proj(self, input, start=0, end=None):
    #     weight = self.in_proj_weight
    #     bias = self.in_proj_bias
    #     weight = weight[start:end, :]
    #     if bias is not None:
    #         bias = bias[start:end]
    #     return F.linear(input, weight, bias)
#######################################################################################################
    # def reorder_incremental_state(self, incremental_state, new_order):
    #     """Reorder buffered internal state (for incremental generation)."""
    #     input_buffer = self._get_input_buffer(incremental_state)
    #     if input_buffer is not None:
    #         for k in input_buffer.keys():
    #             input_buffer[k] = input_buffer[k].index_select(1, new_order)
    #         self._set_input_buffer(incremental_state, input_buffer)
    #
    # def _get_input_buffer(self, incremental_state):
    #     return utils.get_incremental_state(
    #         self,
    #         incremental_state,
    #         'attn_state',
    #     ) or {}
    #
    # def _set_input_buffer(self, incremental_state, buffer):
    #     utils.set_incremental_state(
    #         self,
    #         incremental_state,
    #         'attn_state',
    #         buffer,
    #     )
