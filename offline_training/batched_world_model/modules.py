import re

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


class LSTMDecoder(nn.Module):

    DESCRIPTION_LENGTH = 14

    def __init__(self, tokenizer, hidden_dim, device):

        super().__init__()

        vocab_size = len(tokenizer.get_vocab())
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # tie linear weight with embeddings weight
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.linear.weight = self.embeddings.weight

        self.mask = self._make_mask(tokenizer, vocab_size).to(device)
        self.tokenizer = tokenizer
        self.device = device

    def _make_mask(self, tokenizer, vocab_size):
        mask = torch.zeros((self.DESCRIPTION_LENGTH, vocab_size)).bool()
        for j in range(vocab_size):
            token = tokenizer.id_to_token(j)
            for i in range(4):
                if ('%d_' % i) not in token or 'row_' not in token:
                    mask[i * 2, j] = 1
            for i in range(4):
                if ('%d_' % i) not in token or 'col_' not in token:
                    mask[i * 2 + 1, j] = 1
            for i in range(8, 12):
                if ('%d_' % (i - 8)) not in token or 'id_' not in token:
                    mask[i, j] = 1
            if 'reward' not in token:
                mask[12, j] = 1
            if 'done' not in token:
                mask[13, j] = 1
        return mask

    def decode(self, init_state, tokens):
        state = (init_state.unsqueeze(0), init_state.unsqueeze(0))
        embed = self.embeddings(tokens)
        hidden, _ = self.lstm(embed, state)
        logit = self.linear(hidden)
        logit.masked_fill_(self.mask, float('-inf'))
        return logit

    def generate(self, init_state):

        token = torch.ones(init_state.shape[0], device=self.device).long() * \
            self.tokenizer.token_to_id('_start_')
        embed = self.embeddings(token)
        state = (init_state.unsqueeze(0), init_state.unsqueeze(0))

        pred_seq = []
        for i in range(self.DESCRIPTION_LENGTH):

            hidden, state = self.lstm(embed.unsqueeze(1), state)
            hidden = hidden.squeeze(1)
            logit = self.linear(hidden)
            logit.masked_fill_(self.mask[i], float('-inf'))

            """
            one_hot = F.gumbel_softmax(logit, hard=True)
            token = one_hot.max(dim=-1)[1]
            pred_seq.append(token.unsqueeze(1))
            embed = F.linear(one_hot, self.embeddings.weight.t())
            """

            dist = torch.distributions.Categorical(logits=logit)
            token = dist.sample()
            pred_seq.append(token.unsqueeze(1))

            embed = self.embeddings(token)

        pred_seq = torch.cat(pred_seq, dim=1)

        return pred_seq
