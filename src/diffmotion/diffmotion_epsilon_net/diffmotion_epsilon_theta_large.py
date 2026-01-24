import torch
from torch import nn, Tensor
from src import utils
import torch.nn.functional as F
from src.diffmotion.components.conformer_block import ConformerBlock
from src.diffmotion.components.conformer.convolution import PointwiseConv1d, DepthwiseConv1d
from src.diffmotion.components.conformer.activation import Swish, GLU
from src.diffmotion.components.conformer.embedding import PositionalEncoding

log = utils.get_pylogger(__name__)


class NoScaleDropout(nn.Module):
    """
        Dropout without rescaling and variable dropout rates.
    """

    def __init__(self, rate_max) -> None:
        super().__init__()
        self.rate_max = rate_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate_max == 0:
            # log.info('Drop out in not training!')
            return x
        else:
            # log.info('Drop out in training!')
            rate = torch.empty(1, device=x.device).uniform_(0, self.rate_max)
            mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - rate)
            return x * mask


def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift


def _build_embedding(dim, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
    table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=5000):  # 5000
        super().__init__()
        self.register_buffer(
            "embedding", _build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels, motion_decoder=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output = motion_decoder
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        # x = self.linear(x)
        # x = self.norm_final(x)
        x = self.output(x)
        return x


class TrinityEpsilonTheta(nn.Module):
    def __init__(self,
                 encoder_dim: int = 512,
                 target_dim: int = 65,
                 time_emb_dim: int = 512,
                 atten_sel: str = 'conformer',  # ''full_transformer'
                 mask_sel: str = 'causalMask',  # 'tridiagonal_matrix', 'no_mask'
                 separate_wavlm: bool = False,
                 wavlm_layer: int = 12,
                 causal_mask_diagonal: int = 0,
                 upper_offset: int = 20,
                 lower_offset: int = -20,
                 position_embedding_type: str = 'Transformer_FX',  # 'TISA'( translation-invariant self-attention)
                 condition_strategy: str = 'adaptive_layer_norm',
                 block_depth: int = 4,
                 # Conformer parameters
                 num_att_heads: int = 8,
                 attention_dropout_p: float = 0.1,
                 conv_kernel_size: int = 3,
                 feed_forward_expansion_factor: int = 4,
                 feed_forward_dropout_p: float = 0.1,
                 mlp_ratio: float = 4.0,
                 motion_encoder: nn = None,
                 wavLMEncoder: nn = None,
                 motion_decoder: nn = None,
                 cond_dropout_rate: float = 0,
                 conv_depthwise: bool = False,
                 style_encode: bool = True,
                 use_DropKey: bool = False,
                 mask_ratio: float = 0.3,
                 # cond_dropout_noscale: bool =  True,
                 # cond_dropout_ratemax: float = 0.4,
                 ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.time_emb_dim = time_emb_dim
        self.atten_sel = atten_sel
        self.mask_selection = mask_sel
        self.position_embedding_type = position_embedding_type
        self.model_block = None
        self.motion_encoder = motion_encoder
        self.wavLMEncoder = wavLMEncoder
        self.motion_decoder = motion_decoder
        self.separate_wavlm = separate_wavlm
        self.wavlm_layer = wavlm_layer
        self.style_encode = style_encode
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio
        ##################### Sytle Encoder #######################################
        if self.style_encode is True:
            if not conv_depthwise:
                self.style_encoder = nn.Sequential(
                    Transpose(shape=(1, 2)),
                    nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=500, stride=1, dilation=2),
                    Transpose(shape=(1, 2)),
                    nn.LayerNorm(1024),
                )  # the numbers of parameters in Conv1D
            elif conv_depthwise:
                log.info('Style_encoder Depthwise!!!')
                self.style_encoder = nn.Sequential(
                    nn.LayerNorm(1024),
                    Transpose(shape=(1, 2)),
                    PointwiseConv1d(1024, 1024 * 2, stride=1, padding=0, bias=True),
                    GLU(dim=1),
                    DepthwiseConv1d(1024, 1024, 999, stride=1, padding=0, bias=False),
                    nn.BatchNorm1d(1024),
                    Swish(),
                    PointwiseConv1d(1024, 1024, stride=1, padding=0, bias=True),
                    nn.Dropout(p=0.2),
                    Transpose(shape=(1, 2)),
                )  # the numbers of parameters in Conv1D
        self.dropout = nn.Dropout(p=cond_dropout_rate)
        ############################### Down_Sampler #######################################################
        self.down_sampler = nn.Sequential(
            Transpose(shape=(1, 2)),
            # for base wavlm model
            nn.Conv1d(in_channels=1024, out_channels=encoder_dim, kernel_size=201, stride=2, dilation=1),  # 20s
            # nn.Conv1d(in_channels=768, out_channels=encoder_dim, kernel_size=41, stride=2, dilation=1),                 # 20s 24fps
            # 60 = d * (k - 1)
            Transpose(shape=(1, 2)),
            nn.LayerNorm(encoder_dim),
            nn.LeakyReLU(0.2, True),
        )

        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=encoder_dim
        )
        self.blocks = nn.ModuleList([])
        if use_DropKey:
            total_mask_ratio = self.mask_ratio
            decrease_step = total_mask_ratio / block_depth
        for i in range(block_depth):
            self.blocks.append(ConformerBlock(hidden_size=encoder_dim,
                                              mlp_ratio=mlp_ratio,
                                              num_attention_heads=num_att_heads,
                                              attention_dropout_p=attention_dropout_p,
                                              conv_kernel_size=conv_kernel_size,
                                              feed_forward_expansion_factor=feed_forward_expansion_factor,
                                              feed_forward_dropout_p=feed_forward_dropout_p,
                                              mask_selection=mask_sel,
                                              position_embedding_type=position_embedding_type,
                                              condition_strategy=condition_strategy,
                                              causal_mask_diagonal=causal_mask_diagonal,
                                              upper_offset=upper_offset,
                                              lower_offset=lower_offset,
                                              atten_sel=self.atten_sel,
                                              use_DropKey=self.use_DropKey,
                                              mask_ratio=self.mask_ratio,
                                              ))
            if use_DropKey:
                self.mask_ratio -= decrease_step
        self.final_layer = FinalLayer(encoder_dim, target_dim, motion_decoder=motion_decoder)
        # self.c = None
        self.wav_encode = None

    def WavLM_Encoder(self, cond, style_encode):
        _, all_wavlm_layer = self.wavLMEncoder(cond)  # [64,299,768]
        wavlm_cond = all_wavlm_layer[self.wavlm_layer]
        if style_encode is True:
            style_encoded = self.style_encoder(wavlm_cond)
            wavlm_cond_style_embedded = wavlm_cond + style_encoded
        else:
            wavlm_cond_style_embedded = wavlm_cond
        wavlm_cond_style_embedded = self.down_sampler(wavlm_cond_style_embedded)  # [64,58,512]
        wavlm_cond_style_embedded = self.dropout(wavlm_cond_style_embedded)
        # count = (wavlm_cond_style_embedded == 0).sum().item()
        return wavlm_cond_style_embedded

    def forward(self, inputs, cond, time, processing_state='training', last_time_stamp=999):
        """
        inputs: [B, T, D] tensor of gestures seq
        condition: [B, T, D] tensor of control seq, such as audio features
        dif_time_step: [N,] tensor of diffusion timesteps
        """
        t = self.diffusion_embedding(time.type(torch.long)).unsqueeze(1)

        x = self.motion_encoder(inputs)  # [64,58,512]
        if processing_state == 'training' or processing_state == 'validation' or (
                processing_state == 'test' and (last_time_stamp in time)):
        # if  self.wav_encode is None:
            # log.info("wavLM_Encoding!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.wav_encode = self.WavLM_Encoder(cond=cond, style_encode=self.style_encode)
        c = t + self.wav_encode
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return x
