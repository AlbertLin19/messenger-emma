import math
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from offline_training.batched_world_model.modules import BatchedEncoder, BatchedDecoder
from offline_training.batched_world_model.utils import batched_convert_grid_to_multilabel, batched_convert_multilabel_to_emb, batched_convert_prob_to_multilabel

from messenger.envs.config import NPCS, NO_MESSAGE, WITH_MESSAGE

ENTITY_IDS = {entity.name: entity.id for entity in NPCS}
MOVEMENT_TYPES = {
    "chaser": 0,
    "fleeing": 1,
    "immovable": 2,
}
ROLE_TYPES = {
    "message": 1,
    "goal": 2,
    "enemy": 0,
}

ROLE_ORDER = ['enemy', 'message', 'goal']

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

class BatchedWorldModel(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.grid_channels = 5
        self.batch_size = args.batch_size
        self.device = args.device

        self.hidden_size = hidden_size = args.hidden_size
        attr_embed_dim = args.attr_embed_dim
        action_embed_dim = args.action_embed_dim

        self.role_embeddings = nn.Embedding(len(ROLE_TYPES) + 2, attr_embed_dim)
        self.movement_embeddings = nn.Embedding(len(MOVEMENT_TYPES) + 2, attr_embed_dim)

        conv_params = {
            'kernel': [3, 3, 3],
            'stride': [1, 2, 2],
            'padding': [2, 1, 1],
            'in_channels' : [attr_embed_dim, 64, 64],
            'hidden_channels' : [64, 64, 64],
            'out_channels' : [64, 64, 64],
        }
        F = conv_params['kernel']
        S = conv_params['stride']
        P = conv_params['padding']
        in_channels = conv_params['in_channels']
        out_channels = conv_params['out_channels']
        hidden_channels = conv_params['hidden_channels']

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
        self.encoder = nn.Sequential(*encoder_layers)
        test_in = torch.zeros((1, attr_embed_dim, 10, 10))
        test_out = self.encoder(test_in)
        self.after_encoder_shape = test_out.shape

        enc_dim = math.prod(self.after_encoder_shape[1:])
        self.before_lstm_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_dim, hidden_size),
            nn.ReLU()
        )

        self.action_embeddings = nn.Embedding(5, action_embed_dim)
        self.lstm = nn.LSTM(
            hidden_size + action_embed_dim,
            hidden_size,
        )

        self.after_lstm_projector = nn.Sequential(
            nn.Linear(hidden_size, enc_dim),
            nn.ReLU()
        )
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

        self.decoder = nn.Sequential(
            *decoder_layers,
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels[0],
                out_channels=self.grid_channels,
                kernel_size=1
            )
        )

        nonexistence_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ]
        for _ in range(len(in_channels) - 1):
            nonexistence_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])
        nonexistence_layers.append(nn.Linear(hidden_size, self.grid_channels))
        self.nonexistence_head = nn.Sequential(*nonexistence_layers)

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Flatten(start_dim=0, end_dim=-1),
        )

        self.done_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Flatten(start_dim=0, end_dim=-1),
        )

        # training parameters
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.reward_loss_weight = args.reward_loss_weight
        self.done_loss_weight = args.done_loss_weight

        # loss accumulation
        self.real_grid_loss_total = 0
        self.real_reward_loss_total = 0
        self.real_done_loss_total = 0
        self.real_backprop_count = 0

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, grids, manuals, ground_truths, actions, lstm_states):

        movements = []
        roles = []
        for triplet in ground_truths:
            movements.append([MOVEMENT_TYPES[e[1]] for e in triplet] + [3, 4])
            roles.append([ROLE_TYPES[e[2]] for e in triplet] + [3, 4])

        movements = torch.tensor(movements).to(self.device)
        roles = torch.tensor(roles).to(self.device)

        b, h, w, c = grids.shape
        movements = movements.view(b, 1, 1, c).repeat(1, h, w, 1)
        movements_embed = self._select(self.movement_embeddings, movements)

        roles = roles.view(b, 1, 1, c).repeat(1, h, w, 1)
        roles_embed = self._select(self.role_embeddings, roles)

        mask = grids.view(b, h, w, c, 1).repeat(1, 1, 1, 1, movements_embed.shape[-1])
        # b x h x w x c x embed_dim
        movements_embed = movements_embed * mask
        roles_embed = roles_embed * mask
        # b x h x w x embed_dim
        movements_embed = movements_embed.sum(dim=-2)
        roles_embed = roles_embed.sum(dim=-2)
        movements_embed = movements_embed.permute((0, 3, 1, 2))
        roles_embed = roles_embed.permute((0, 3, 1, 2))

        grids_embed = movements_embed + roles_embed

        latents = self.encoder(grids_embed)
        latents = self.before_lstm_projector(latents)

        actions = self.action_embeddings(actions)

        mem_ins = torch.cat((latents, actions), dim=-1).unsqueeze(0)
        mem_outs, (hidden_states, cell_states) = self.lstm(mem_ins, lstm_states)
        mem_outs = mem_outs.squeeze(0)

        decoder_inps = self.after_lstm_projector(mem_outs)
        decoder_inps = decoder_inps.view(decoder_inps.shape[0], *self.after_encoder_shape[1:])

        pred_grid_logits = self.decoder(decoder_inps)
        pred_nonexistence_logits = self.nonexistence_head(mem_outs)

        pred_rewards = self.reward_head(mem_outs)
        pred_done_logits = self.done_head(mem_outs)

        return (pred_grid_logits, pred_nonexistence_logits, pred_rewards, pred_done_logits), (hidden_states, cell_states)

    def create_loc_logits_and_probs(self, grid_logits, nonexistence_logits, grid_probs):
        grid_logits = grid_logits.view(grid_logits.shape[0], grid_logits.shape[1], -1)
        nonexistence_logits = nonexistence_logits.unsqueeze(-1)
        location_logits = torch.cat((grid_logits, nonexistence_logits), dim=-1)

        grid_probs = grid_probs.permute((0, 3, 1, 2))
        grid_probs = grid_probs.view(grid_probs.shape[0], grid_probs.shape[1], -1)
        nonexistence_probs = (1. - grid_probs.sum(dim=-1)).unsqueeze(-1)
        location_probs = torch.cat((grid_probs, nonexistence_probs), dim=-1)

        return location_logits, location_probs

    def grid_loss(self, location_logits, location_probs):
        location_logits = location_logits.view(-1, location_logits.shape[-1])
        location_probs  = location_probs.view(-1, location_probs.shape[-1])
        loss = F.cross_entropy(location_logits, location_probs, reduction='sum') / (self.batch_size * self.grid_channels)
        return loss

    def reward_loss(self, pred_rewards, rewards):
        return F.mse_loss(pred_rewards, rewards, reduction='sum') / self.batch_size

    def done_loss(self, pred_done_logits, done_probs):
        return F.binary_cross_entropy_with_logits(pred_done_logits, done_probs, reduction='sum') / self.batch_size

    # reset hidden states for real prediction
    def real_state_reset(self, init_grids, idxs=None):
        if idxs is None:
            self.real_hidden_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.real_cell_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
        else:
            self.real_hidden_states[:, idxs] = 0
            self.real_cell_states[:, idxs] = 0

    # detach hidden states for real prediction
    def real_state_detach(self):
        self.real_hidden_states = self.real_hidden_states.detach()
        self.real_cell_states = self.real_cell_states.detach()

    def reorder_ground_truths(self, ground_truths):
        new_ground_truths = []
        for i, triplet in enumerate(ground_truths):
            new_triplet = []
            for r in ROLE_ORDER:
                for e in triplet:
                    if e[2] == r:
                        new_triplet.append(e)
                        break
            new_ground_truths.append(new_triplet)
        return new_ground_truths

    def reformat_grids(self, grids):
        b, h, w, c = grids.shape
        new_grids = torch.zeros((b, h, w, self.grid_channels), device=self.device)
        for i in range(3):
            new_grids[..., i] = (grids[..., i] > 0).float()
        new_grids[..., 3] = (grids[..., 3] == NO_MESSAGE.id).float()
        new_grids[..., 4] = (grids[..., 3] == WITH_MESSAGE.id).float()
        return new_grids

    def probs_to_probs_by_entityid(self, probs, ground_truths):
        b, c, n = probs.shape
        probs_by_entityid = torch.zeros((b, 17, n), device=self.device)
        # default: no entities exist
        probs_by_entityid[:, :, -1] = 1
        for i, triplet in enumerate(ground_truths):
            for j, e in enumerate(triplet):
                e_id = ENTITY_IDS[e[0]]
                probs_by_entityid[i, e_id] = probs[i, j]
            probs_by_entityid[i, NO_MESSAGE.id] = probs[i, 3]
            probs_by_entityid[i, WITH_MESSAGE.id] = probs[i, 4]
        return probs_by_entityid


    # make real prediction and accumulate real loss
    def real_step(self,
            old_grids,
            manuals,
            ground_truths,
            actions,
            grids,
            rewards,
            dones,
            backprop_idxs):

        ground_truths = self.reorder_ground_truths(ground_truths)
        old_grids = self.reformat_grids(old_grids)
        grids = self.reformat_grids(grids)
        done_probs = dones.float()

        ((pred_grid_logits, pred_nonexistence_logits, pred_rewards, pred_done_logits),
        (self.real_hidden_states, self.real_cell_states)) = \
            self.forward(
                    old_grids,
                    manuals,
                    ground_truths,
                    actions,
                    (self.real_hidden_states, self.real_cell_states),
            )

        ground_truths = [ground_truths[i] for i in backprop_idxs.tolist()]
        pred_grid_logits = pred_grid_logits[backprop_idxs]
        pred_nonexistence_logits = pred_nonexistence_logits[backprop_idxs]
        pred_rewards = pred_rewards[backprop_idxs]
        pred_done_logits = pred_done_logits[backprop_idxs]
        grids = grids[backprop_idxs]
        rewards = rewards[backprop_idxs]
        done_probs = done_probs[backprop_idxs]

        pred_loc_logits, loc_probs = self.create_loc_logits_and_probs(
            pred_grid_logits,
            pred_nonexistence_logits,
            grids
        )
        pred_loc_probs = pred_loc_logits.softmax(dim=-1)

        self.real_grid_loss_total += self.grid_loss(pred_loc_logits, loc_probs)
        self.real_reward_loss_total += self.reward_loss(pred_rewards, rewards)
        self.real_done_loss_total += self.done_loss(pred_done_logits, done_probs)

        self.real_backprop_count += 1

        with torch.no_grad():
            pred_loc_probs_by_entityid = self.probs_to_probs_by_entityid(pred_loc_probs, ground_truths)
            loc_probs_by_entityid = self.probs_to_probs_by_entityid(loc_probs, ground_truths)
            pred_done_probs = torch.sigmoid(pred_done_logits)

        return (pred_loc_probs_by_entityid, pred_rewards, pred_done_probs), (loc_probs_by_entityid, rewards, done_probs)

    # update model via real loss
    def real_loss_update(self):
        self.optimizer.zero_grad()
        real_loss_total = self.real_grid_loss_total + \
            self.reward_loss_weight * self.real_reward_loss_total + \
            self.done_loss_weight * self.real_done_loss_total
        real_loss_total.backward()
        self.optimizer.step()

        loss_values = self.real_loss_reset()
        self.real_state_detach()

        return loss_values



    # reset real loss
    def real_loss_reset(self):
        with torch.no_grad():
            real_grid_loss_mean = self.real_grid_loss_total / self.real_backprop_count
            real_reward_loss_mean = self.real_reward_loss_total / self.real_backprop_count
            real_done_loss_mean = self.real_done_loss_total / self.real_backprop_count
            real_loss_mean = (self.real_grid_loss_total + \
                    self.reward_loss_weight * self.real_reward_loss_total + \
                    self.done_loss_weight * self.real_done_loss_total) / self.real_backprop_count

        self.real_grid_loss_total = 0
        self.real_reward_loss_total = 0
        self.real_done_loss_total = 0
        self.real_backprop_count = 0

        return real_grid_loss_mean.item(), \
               real_reward_loss_mean.item(), \
               real_done_loss_mean.item(), \
               real_loss_mean.item()


