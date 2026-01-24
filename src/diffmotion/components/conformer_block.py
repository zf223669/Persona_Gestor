import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from src.diffmotion.components.conformer.attention import MultiHeadedSelfAttentionModule
from src.diffmotion.components.conformer.convolution import ConformerConvModule
from src.diffmotion.components.conformer.feed_forward import FeedForwardModule
from src.diffmotion.components.conformer.modules import ResidualConnectionModule
from src import utils
from src.diffmotion.components.conformer.CrossAttention import CrossAttention

log = utils.get_pylogger(__name__)


def adaptive_modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift


class ConformerBlock(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 mlp_ratio: float = 4.0,
                 feed_forward_expansion_factor: int = 4,
                 feed_forward_dropout_p: float = 0.1,
                 num_attention_heads: int = 8,
                 attention_dropout_p: float = 0.1,
                 conv_kernel_size: int = 3,
                 conv_expansion_factor: int = 2,
                 conv_dropout_p: float = 0.1,
                 mask_selection: str = 'causalMask',
                 position_embedding: str = 'Transformer_FX',
                 condition_strategy: str = 'adaptive_layer_norm',
                 causal_mask_diagonal: int = 0,
                 upper_offset: int = 20,
                 lower_offset: int = -20,
                 atten_sel: str = 'conformer',
                 use_DropKey: bool = False,
                 mask_ratio: float = 0.3,
                 **block_kwargs):
        super().__init__()
        self.mask_selection = mask_selection
        self.position_embedding = position_embedding
        self.condition_strategy = condition_strategy
        self.causal_mask_diagonal = causal_mask_diagonal
        self.upper_offset = upper_offset
        self.lower_offset = lower_offset
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.atten_sel = atten_sel
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio

        self.attn = nn.Sequential(
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=hidden_size,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                    mask_selection=self.mask_selection,
                    causal_mask_diagonal=self.causal_mask_diagonal,  # for causal mask
                    upper_offset=self.upper_offset,  # for diagonal mask
                    lower_offset=self.lower_offset,
                    atten_sel=self.atten_sel,
                    use_DropKey=self.use_DropKey,
                    mask_ratio=self.mask_ratio,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=hidden_size,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if self.condition_strategy == 'adaptive_layer_norm':            ### if condition_strategy != aln, have to comment it,bug
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )
        elif self.condition_strategy == 'cross_attention':
            log.info('cross_attention!!!!!!!!!!')
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.condition_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

            self.cross_atten = CrossAttention(dim=hidden_size, heads=num_attention_heads)
        elif self.condition_strategy == 'in_context_conditioning':
            log.info('in_context_conditioning!!!!!!!!!!')
        elif self.condition_strategy == 'concat_feature':
            log.info('concat_feature!!!!!!!!!!')


    def forward(self, x, c):
        if self.condition_strategy == 'adaptive_layer_norm':
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
            x = x + gate_msa * self.attn(adaptive_modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp * self.mlp(adaptive_modulate(self.norm2(x), shift_mlp, scale_mlp))
        elif self.condition_strategy == 'cross_attention':
            # log.info('cross_attention')
            x = x + self.attn(self.norm1(x))
            x = x + self.cross_atten(self.norm2(x), self.condition_norm(c), self.condition_norm(c))
            x = x + self.mlp(self.norm3(x))
        elif self.condition_strategy == 'in_context_conditioning':
            # log.info('in_context_conditioning')
            x = x + c
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        elif self.condition_strategy == 'concat_feature':
            x = x + self.attn(x)
            x = x + self.mlp(self.norm1(x))
        return x
