import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchedEncoder(nn.Module):
    def __init__(self, channel_size, latent_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=latent_size, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=latent_size, out_channels=latent_size, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=latent_size, out_features=latent_size),
        )

    def forward(self, x):
        return self.network(x)

class BatchedDecoder(nn.Module):
    def __init__(self, channel_size, latent_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=latent_size),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(-1, 1, 1)),
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=latent_size, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=channel_size, kernel_size=4, stride=2),
        )

    def forward(self, x):
        return self.network(x)


class MyGroupNorm(nn.Module):

    # num_channels: num_groups
    GROUP_NORM_LOOKUP = {
        16: 2,  # -> channels per group: 8
        32: 4,  # -> channels per group: 8
        64: 8,  # -> channels per group: 8
        128: 8,  # -> channels per group: 16
        256: 16,  # -> channels per group: 16
        320: 16,  # -> channels per group: 16
        512: 32,  # -> channels per group: 16
        640: 32,  # -> channels per group: 16
        1024: 32,  # -> channels per group: 32
        2048: 32,  # -> channels per group: 64
    }

    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=self.GROUP_NORM_LOOKUP[num_channels],
                                 num_channels=num_channels)

    def forward(self, x):
        x = self.norm(x)
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)

# from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L2
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        padding=1,
        dilation=0,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

conv_params = {
    'kernel': [3, 3, 3],
    'stride': [1, 2, 2],
    'padding': [2, 1, 1],
    'in_channels' : [-1, 64, 64],
    'hidden_channels' : [64, 64, 64],
    'out_channels' : [64, 64, 64],
}

class ResNetEncoder(nn.Module):

    def __init__(self, in_dim):

        super().__init__()

        F = conv_params['kernel'][:]
        S = conv_params['stride'][:]
        P = conv_params['padding'][:]
        in_channels = conv_params['in_channels'][:]
        out_channels = conv_params['out_channels'][:]
        hidden_channels = conv_params['hidden_channels'][:]

        in_channels[0] = in_dim


        encoder_layers = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=hidden_channels[i],
                    kernel_size=F[i],
                    stride=S[i],
                    padding=P[i],
                ),
                nn.ReLU(),
                BasicBlock(
                    inplanes=hidden_channels[i],
                    planes=out_channels[i],
                    padding=1,
                    norm_layer=MyGroupNorm,
                    downsample=nn.Conv2d(hidden_channels[i], out_channels[i], 1)
                )
            )
            for i in range(len(in_channels))
        ]

        self.num_layers = len(encoder_layers)
        self.model = nn.Sequential(*encoder_layers)

    def forward(self, in_data):
        return self.model(in_data)


class ResNetDecoder(nn.Module):

    def __init__(self, in_dim, out_dim=None):

        super().__init__()

        F = conv_params['kernel'][:]
        S = conv_params['stride'][:]
        P = conv_params['padding'][:]
        in_channels = conv_params['in_channels'][:]
        out_channels = conv_params['out_channels'][:]
        hidden_channels = conv_params['hidden_channels'][:]

        in_channels[0] = in_dim

        decoder_layers = [
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=out_channels[i],
                    out_channels=hidden_channels[i],
                    kernel_size=F[i],
                    stride=S[i],
                    padding=P[i],
                    output_padding=1 if S[i] > 1 else 0
                ),
                nn.ReLU(),
                BasicBlock(
                    inplanes=hidden_channels[i],
                    planes=in_channels[i],
                    padding=1,
                    norm_layer=MyGroupNorm,
                    downsample=nn.Conv2d(hidden_channels[i], in_channels[i], 1)
                )
            )
            for i in reversed(range(len(in_channels)))
        ]

        if out_dim is not None:
            decoder_layers.extend([
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=in_channels[0],
                        out_channels=out_dim,
                        kernel_size=1
                    )
                ]
            )

        self.model = nn.Sequential(*decoder_layers)


    def forward(self, in_data):
        return self.model(in_data)


# https://github.com/sooftware/attentions/blob/master/attentions.py
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value, mask=None, true_attn=None):

        if true_attn is None:
            score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
            if mask is not None:
                score.masked_fill_(mask.view(score.size()), -float('Inf'))
            attn = F.softmax(score, -1)
        else:
            attn = true_attn

        context = torch.bmm(attn, value)
        return context, attn
