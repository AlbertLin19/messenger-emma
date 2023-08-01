"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn

#from dataset import Batch
from .lpips import LPIPS
#from .nets import Encoder, Decoder, EncoderDecoderConfig
#from utils import LossWithIntermediateLosses

from offline_training.batched_world_model.modules import ResNetEncoder, ResNetDecoder

n_grid_channels = 4
n_ids = 17


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):

    def __init__(self, args):

        super().__init__()
        vocab_size = args.codebook_size

        id_embed_dim = args.id_embed_dim
        codebook_embed_dim = args.codebook_embed_dim

        self.id_embedding = nn.Embedding(n_ids, id_embed_dim)
        self.channel_embedding = nn.Embedding(n_grid_channels, id_embed_dim)
        self.encoder, self.decoder = self.make_encoder_decoder(args)
        self.pre_quant_conv = torch.nn.Conv2d(args.z_channels, codebook_embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, codebook_embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(codebook_embed_dim, args.z_channels, 1)

        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if args.with_lpips else None

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    def make_encoder_decoder(self, args):
        conv_params = {
            'kernel': [3, 3, 3],
            'stride': [1, 2, 2],
            'padding': [2, 1, 1],
            'in_channels' : [-1, 64, 64],
            'hidden_channels' : [64, 64, 64],
            'out_channels' : [64, 64, 64],
        }


        id_embed_dim = args.id_embed_dim
        conv_params['out_channels'][-1] = args.z_channels
        encoder = ResNetEncoder(id_embed_dim * 4, conv_params=conv_params)
        decoder = ResNetDecoder(id_embed_dim * 4, conv_params=conv_params, out_dim=n_grid_channels * n_ids)

        return encoder, decoder

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        encoder_output = self.encode(x, should_preprocess)
        decoder_input = encoder_output.z + (encoder_output.z_quantized - encoder_output.z).detach()
        decoder_output = self.decode(decoder_input, should_postprocess)
        decoder_output = decoder_output.permute(0, 2, 3, 1)
        logits = decoder_output.view(*decoder_output.shape[:3], 4, 17)
        return encoder_output, logits

    def compute_loss(self, observations):
        #assert self.lpips is not None
        #observations = self.preprocess_input(rearrange(batch, 'b t c h w -> (b t) c h w'))

        encoder_output, logits = self(observations, should_preprocess=False, should_postprocess=False)
        z = encoder_output.z
        z_quantized = encoder_output.z_quantized

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0

        self.loss = []
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()
        reconstruction_loss = torch.nn.functional.cross_entropy(logits.flatten(0, -2), observations.flatten(), reduction='mean')

        return reconstruction_loss, commitment_loss

    def learn(self, observations, is_eval=False):

        loss = {}
        loss['reconstruction'], loss['commitment'] = self.compute_loss(observations)
        loss['total'] = loss['reconstruction'] + loss['commitment']

        if not is_eval:
            self.optimizer.zero_grad()
            loss['total'].backward()
            self.optimizer.step()

        for k in loss:
            loss[k] = loss[k].item()

        return loss

    def embed_grid(self, x):
        grid_embed = self.id_embedding(x)
        grid_embed = grid_embed.view(*grid_embed.shape[:3], -1)
        grid_embed = grid_embed.permute(0, 3, 1, 2)
        return grid_embed

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        x = self.embed_grid(x)
        shape = x.shape
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
