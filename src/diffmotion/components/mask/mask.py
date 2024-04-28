import torch
from torch import Tensor
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter


# -m model.eps_theta_mod.conv_kernel_size=3 model.eps_theta_mod.mask_sel="dman"
def causal_mask(batch_size: int, seq_length: int, diagonal: int = 0) -> Tensor:
    mask = (torch.triu(torch.ones((seq_length, seq_length), device='cuda'), diagonal=diagonal) == 1).unsqueeze(
        0)  # (batch, 1, time2) or (batch, time1, time2)产生一个上三角矩阵，上三角的值全为0。
    mask = mask.repeat(batch_size, 1, 1)
    return mask


def upper_lower_diagonal_mask(batch_size: int, seq_length: int, upper_offset: int = 20,
                              lower_offset: int = -20) -> Tensor:
    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=400)
    assert upper_offset >= 0
    assert lower_offset <= 0
    upper_mask = (torch.triu(torch.ones((seq_length, seq_length), device='cuda'), diagonal=upper_offset) == 1)
    lower_mask = (torch.tril(torch.ones((seq_length, seq_length), device='cuda'), diagonal=lower_offset) == 1)
    # combine_mask = torch.bitwise_and(upper_mask, lower_mask)
    combine_mask = upper_mask | lower_mask
    # full_mask = torch.ones((seq_length, seq_length), device='cuda')
    # full_mask.masked_fill_(upper_mask, 0).masked_fill_(lower_mask, 0)
    # full_mask = ((1 - full_mask) == 1)
    mask = combine_mask.repeat(batch_size, 1, 1)
    torch.set_printoptions(profile="default")
    return mask


class DMAN(nn.Module):
    def __init__(self, max_len=200, embed_dim=1024, num_heads=16, ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.forward_position_x = Parameter(torch.Tensor(max_len))
        self.backward_position_x = Parameter(torch.Tensor(max_len))
        self.head_x = Parameter(torch.Tensor(self.num_heads))
        self.context_x_fc = nn.Linear(self.embed_dim, 1, False)
        self.indices = torch.arange(max_len, dtype=torch.int32, device=torch.device('cuda')).unsqueeze(0). \
            add(torch.arange(0, max_len + 1, dtype=torch.int32, device=torch.device('cuda')).flip([0]).unsqueeze(1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.forward_position_x is not None:
            nn.init.normal_(self.forward_position_x)
        if self.backward_position_x is not None:
            nn.init.normal_(self.backward_position_x)
        if self.head_x is not None:
            nn.init.normal_(self.head_x)
        if self.context_x_fc is not None:
            nn.init.xavier_uniform_(self.context_x_fc.weight)

    def forward(self, **kw):
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

        # indices = self.indices[:key_len, :key_len].cuda()
        indices = self.indices[:key_len, :key_len]

        position_x = torch.take(position_x, indices.long())
        position_x = position_x.view(key_len, key_len)

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


def test_mask(size: int = 10):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask


if __name__ == "__main__":
    DMAN_module = DMAN(max_len=120, num_heads=16, )
    inputs = torch.rand((5, 120, 1024))
    mask = DMAN_module(context=inputs, key_len=50, bsz=5)
    print(mask)
    # test_mask()
    # causal_mask(10, 10)
    # upper_lower_diagonal_mask(10, 10, 3, -3)
