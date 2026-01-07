import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter
import math
from typing import Tuple
from src.diffmotion.components.conformer.activation import Swish, GLU
from src.diffmotion.components.conformer.convolution import PointwiseConv1d, DepthwiseConv1d


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class MultiConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, encoder_kernel_size: int = 3, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, ),
            Transpose(shape=(1, 2)),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2, True),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=encoder_kernel_size, stride=1,
                      padding=1, padding_mode='replicate'),
            Transpose(shape=(1, 2)),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2, True),

            # Transpose(shape=(1, 2)),
            # nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1,),
            # Transpose(shape=(1, 2)),
            # nn.LayerNorm(out_channels),
            # nn.LeakyReLU(0.2, True),

            # Add Conv1d
            # nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            # Transpose(shape=(1, 2)),
        )

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(inputs)
        return outputs


class Conv1dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of seqduence output lengths
    """

    def __init__(self, in_channels: int, out_channels: int, encoder_kernel_size: int = 3, batch_norm=False, pca_dim:int=256, patch_size:int = 4 ) -> None:
        super(Conv1dSubampling, self).__init__()
        self.pca_dim = pca_dim
        self.patch_size = patch_size         # 4 * 4 * 156 = 524
        if encoder_kernel_size == 1:
            self.sequential = nn.Sequential(
                Transpose(shape=(1, 2)),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=encoder_kernel_size),
                Transpose(shape=(1, 2)),
            )
        elif encoder_kernel_size == 3:
            self.sequential = nn.Sequential(
                Transpose(shape=(1, 2)),
                # nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=encoder_kernel_size, stride=1,
                #           padding=1, padding_mode='replicate'),
                # Add Conv1d
                ## nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
                # Patch
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=encoder_kernel_size, stride=1,
                          padding=1, padding_mode='replicate'),
                # nn.LayerNorm(out_channels),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=self.patch_size, stride=self.patch_size, bias=False),
                # nn.Conv1d(self.pca_dim, out_channels, kernel_size=1, stride=1, bias=True),
                Transpose(shape=(1, 2)),
            )
        elif encoder_kernel_size == 4:
            self.sequential = nn.Sequential(
                Transpose(shape=(1, 2)),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=encoder_kernel_size,
                          stride=2),
                Transpose(shape=(1, 2)),
            )
        elif encoder_kernel_size == 5:
            self.sequential = nn.Sequential(
                Transpose(shape=(1, 2)),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=encoder_kernel_size,
                          stride=1, padding=2, padding_mode='replicate'),
                Transpose(shape=(1, 2)),
            )
        self.in_channels = in_channels
        self.batch_norm = batch_norm
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(inputs)
        if not self.batch_norm:
            nn.LayerNorm(self.in_channels)
        else:
            nn.BatchNorm1d(self.in_channels)
        nn.Dropout(p=0.1)
        # outputs = F.leaky_relu(outputs, 0.4)
        return outputs


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=1, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding,
                           padding_mode='replicate')
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net


class PoseEncoderConv(nn.Module):
    def __init__(self, gesture_features, encoder_dim, encoder_kernel_size=1, batch_norm=False, simple_encoder=True,
                 encoder_type='single_conv',pca_dim=256,patch_size=4):
        super().__init__()
        print(f'Encoder Type: {encoder_type}')
        self.pca_dim=pca_dim
        self.patch_size = patch_size
        if encoder_type == 'stack_conv':
            self.net = nn.Sequential(
                ConvNormRelu(gesture_features, encoder_dim // 2, batchnorm=True),
                ConvNormRelu(encoder_dim // 2, encoder_dim, batchnorm=True),
                ConvNormRelu(encoder_dim, encoder_dim, True, batchnorm=True),
                nn.Conv1d(encoder_dim, encoder_dim // 2, 3)
            )
        elif encoder_type == 'single_conv':
            self.net = nn.Sequential(
                Conv1dSubampling(in_channels=gesture_features, out_channels=encoder_dim,
                                 encoder_kernel_size=encoder_kernel_size, batch_norm=batch_norm,pca_dim=pca_dim,patch_size=patch_size),
            )
        elif encoder_type == 'separable_conv':
            self.net = nn.Sequential(
                nn.LayerNorm(gesture_features),
                Transpose(shape=(1, 2)),
                nn.Conv1d(gesture_features, gesture_features, kernel_size=encoder_kernel_size, groups=gesture_features,
                          stride=1,
                          padding=1, padding_mode='replicate', bias=True),
                nn.Conv1d(gesture_features, encoder_dim, kernel_size=1, bias=True),
                Transpose(shape=(1, 2)),
            )
        elif encoder_type == 'multi_conv':
            self.net = nn.Sequential(MultiConv(in_channels=gesture_features, out_channels=encoder_dim,
                                               encoder_kernel_size=encoder_kernel_size))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, poses):
        # encode
        out = self.net(poses)
        return out


class PoseDecoderConv(nn.Module):
    def __init__(self, pose_dim, latent_dim, decoder_type, use_convtranspose=False,patch_size = 4):
        super().__init__()
        if decoder_type == 'single_conv' or 'multi_conv':
            if not use_convtranspose:
                self.net = nn.Sequential(
                    Transpose(shape=(1, 2)),
                    nn.Conv1d(in_channels=latent_dim, out_channels=patch_size*pose_dim, kernel_size=1),
                    Transpose(shape=(1, 2)),
                )
            elif use_convtranspose:  # not good
                self.net = nn.Sequential(
                    Transpose(shape=(1, 2)),
                    nn.ConvTranspose1d(in_channels=latent_dim, out_channels=pose_dim, kernel_size=1, padding=0),
                    Transpose(shape=(1, 2)),
                )
        elif decoder_type == 'separable_conv':
            self.net = nn.Sequential(
                nn.LayerNorm(latent_dim),
                Transpose(shape=(1, 2)),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=1, groups=latent_dim, bias=True),
                nn.Conv1d(latent_dim, pose_dim, kernel_size=1, bias=True),
                Transpose(shape=(1, 2)),
            )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, feat, pre_poses=None):
        out = self.net(feat)
        return out


class MotionAE(nn.Module):
    def __init__(self, pose_dim, latent_dim):
        super(MotionAE, self).__init__()

        self.encoder = PoseEncoderConv(34, pose_dim, latent_dim)
        self.decoder = PoseDecoderConv(34, pose_dim, latent_dim)

    def forward(self, pose):
        pose = pose.view(pose.size(0), pose.size(1), -1)
        z = self.encoder(pose)
        pred = self.decoder(z)

        return pred, z


if __name__ == '__main__':
    motion_vae = MotionAE(126, 128)
    pose_1 = torch.rand(4, 34, 126)
    pose_gt = torch.rand(4, 34, 126)

    pred, z = motion_vae(pose_1)
    loss_fn = nn.MSELoss()
    print(f'z.shape: {z.shape}')
    print(f'pred.shape: {pred.shape}')
    print(loss_fn(pose_gt, pred))
