import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        print('CrossAttention')
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, mask=None):
        b, n, _, h = *queries.shape, self.heads
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        queries = queries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b, n, h, -1).transpose(1, 2)
        values = values.view(b, n, h, -1).transpose(1, 2)
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return out
