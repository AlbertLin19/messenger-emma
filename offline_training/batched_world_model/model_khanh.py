import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from offline_training.batched_world_model.modules import BatchedEncoder, BatchedDecoder
from offline_training.batched_world_model.utils import batched_convert_grid_to_multilabel, batched_convert_multilabel_to_emb, batched_convert_prob_to_multilabel

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
        512: 32,  # -> channels per group: 16
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

    def __init__(self, key_type=None,
                       key_dim=None,
                       val_type=None,
                       val_dim=None,
                       memory_type=None,
                       latent_size=None,
                       hidden_size=None,
                       batch_size=None,
                       learning_rate=None,
                       weight_decay=None,
                       reward_loss_weight=None,
                       done_loss_weight=None,
                       prediction_type=None,
                       pred_multilabel_threshold=None,
                       refine_pred_multilabel=None,
                       dropout_prob=None,
                       dropout_loc=None,
                       shuffle_ids=None,
                       attr_embed_dim=None,
                       action_embed_dim=None,
                       device=None):

        super().__init__()

        # world model parameters
        #self.latent_size = latent_size
        #self.memory_type = memory_type
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # text module parameters
        #self.key_type = key_type
        #self.val_type = val_type
        #self.val_dim = val_dim

        """
        if key_type == "oracle":
            # entity query vector
            self.sprite_emb = lambda x: F.one_hot(x, num_classes=17).float()

        elif key_type == "emma":
            # entity query vector
            self.sprite_emb = nn.Embedding(17, key_dim, padding_idx=0).to(device)
            self.attn_scale = np.sqrt(key_dim)

            # descriptor key module
            self.txt_key = nn.Linear(768, key_dim).to(device) # token embedding -> key embedding
            self.scale_key = nn.Sequential(                   # linear combination weights for key embeddings
                nn.Linear(768, 1),
                nn.Softmax(dim=-2)
            ).to(device)

        elif key_type == "emma-mlp_scale":
            # entity query vector
            self.sprite_emb = nn.Embedding(17, key_dim, padding_idx=0).to(device)
            self.attn_scale = np.sqrt(key_dim)

            # descriptor key module
            self.txt_key = nn.Linear(768, key_dim).to(device) # token embedding -> key embedding
            self.scale_key = nn.Sequential(                   # linear combination weights for key embeddings
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Linear(384, 1),
                nn.Softmax(dim=-2)
            ).to(device)

        else:
            raise NotImplementedError

        if val_type == "oracle":
            # avatar value embeddings
            self.avatar_no_message_val_emb = torch.tensor([0, 0, 0, 0, 0, 0, 1], device=device)
            self.avatar_with_message_val_emb = torch.tensor([0, 0, 0, 0, 0, 0, 1], device=device)
        elif val_type == "emma":
            # avatar value embeddings
            self.avatar_no_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))
            self.avatar_with_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))

            # descriptor value module
            self.txt_val = nn.Linear(768, val_dim).to(device) # token embedding -> value embedding
            self.scale_val = nn.Sequential(                   # linear combination weights for value embeddings
                nn.Linear(768, 1),
                nn.Softmax(dim=-2)
            ).to(device)

        elif val_type == "emma-mlp_scale":
            # avatar value embeddings
            self.avatar_no_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))
            self.avatar_with_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))

            # descriptor value module
            self.txt_val = nn.Linear(768, val_dim).to(device) # token embedding -> value embedding
            self.scale_val = nn.Sequential(                   # linear combination weights for value embeddings
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Linear(384, 1),
                nn.Softmax(dim=-2)
            ).to(device)

        elif val_type == "none":
            pass

        else:
            raise NotImplementedError

        self.prediction_type = prediction_type

        # i.e. grid(i, j) is probability that sprite type j exists at location i (independently of other (i, j))
        if self.prediction_type == "existence":
            self.pos_weight = 10*torch.ones(17, device=device) # loss weighting for existence of sprites, reweighted due to large bias of nonexistence of sprites
            self.pos_weight[0] = 3 / 100 # empty sprite, reweighted due to large bias of existence of empty sprite
            self.relevant_cls_idxs = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], device=device) # which sprites to use for eval metrics

        # i.e. grid(i, j) is probability that location i is classified as sprite type j (grid(i, :) sums to 1)
        elif self.prediction_type == "class":
            self.cls_weight = torch.ones(17, device=device) # loss weighting for classification of sprite types
            self.cls_weight[0] = 3 / 100 # empty type, reweighted due to large bias of empty cell type
            self.relevant_cls_idxs = torch.tensor([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], device=device) # which sprites to use for eval metrics

        # i.e. grid(i, j) is probability that sprite type j is at location i (grid(:, j) sums to 1)
        elif self.prediction_type == "location":
            self.loc_weight = torch.ones(101, device=device) # loss reweighting for locations, no reweighting
            self.relevant_cls_idxs = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], device=device) # which sprites to use for eval metrics

        else:
            raise NotImplementedError
        """

        self.role_embeddings = nn.Embedding(len(ROLE_TYPES) + 1, attr_embed_dim)
        self.movement_embeddings = nn.Embedding(len(MOVEMENT_TYPES) + 1, attr_embed_dim)

        conv_params = {
            'kernel': [3, 3],
            'stride': [1, 2],
            'padding': [2, 1],
            'in_channels' : [attr_embed_dim * 4, 64],
            'hidden_channels' : [64, 64],
            'out_channels' : [64, 64],
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
        test_in = torch.zeros((1, attr_embed_dim * 4, 10, 10))
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
                out_channels=4,
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
        nonexistence_layers.append(nn.Linear(hidden_size, 4))
        self.nonexistence_head = nn.Sequential(*nonexistence_layers)

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Flatten(start_dim=0, end_dim=-1),
        ).to(device)

        self.done_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Flatten(start_dim=0, end_dim=-1),
        ).to(device)


        """
        #emb_dim = val_dim + len(self.relevant_cls_idxs) # size of input to encoder

        # input -> latent
        self.encoder = nn.Sequential(
            #nn.Dropout(p=dropout_prob if "input" in dropout_loc else 0),
            BatchedEncoder(embed_dim * 4, ),
            #nn.Dropout(p=dropout_prob if "network" in dropout_loc else 0),
        ).to(device)

        # latent + action -> hidden
        if self.memory_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear( + 5, hidden_size),
                nn.ReLU(),
            ).to(device)
        elif self.memory_type == "lstm":
            self.lstm = nn.LSTM( + 5, hidden_size).to(device)
        else:
            raise NotImplementedError

        # hidden -> latent
        self.projection = nn.Linear(in_features=hidden_size, out_features=).to(device)

        # latent -> grid output
        if prediction_type == "location":
            self.nonexistence = nn.Linear(in_features=, out_features=17).to(device)
        self.decoder = BatchedDecoder(emb_dim, ).to(device)
        self.detector = nn.Sequential(
            nn.Conv2d(in_channels=emb_dim, out_channels=(emb_dim + 17) // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=(emb_dim + 17) // 2, out_channels=17, kernel_size=1, stride=1),
        ).to(device)
        """

        # training parameters
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.reward_loss_weight = reward_loss_weight
        self.done_loss_weight = done_loss_weight

        # loss accumulation
        self.real_grid_loss_total = 0
        self.real_reward_loss_total = 0
        self.real_done_loss_total = 0
        self.real_backprop_count = 0
        #self.imag_grid_loss_total = 0
        #self.imag_reward_loss_total = 0
        #self.imag_done_loss_total = 0
        #self.imag_backprop_count = 0

        # training and prediction args
        self.pred_multilabel_threshold = pred_multilabel_threshold
        self.refine_pred_multilabel = refine_pred_multilabel
        #self.dropout_prob = dropout_prob
        #self.dropout_loc = dropout_loc
        #self.shuffle_ids = shuffle_ids
        self.device = device

    def freeze_key(self):
        for param in self.sprite_emb.parameters():
            param.requires_grad = False
        for param in self.txt_key.parameters():
            param.requires_grad = False
        for param in self.scale_key.parameters():
            param.requires_grad = False

    def unfreeze_key(self):
        for param in self.sprite_emb.parameters():
            param.requires_grad = True
        for param in self.txt_key.parameters():
            param.requires_grad = True
        for param in self.scale_key.parameters():
            param.requires_grad = True

    def freeze_val(self):
        for param in self.txt_val.parameters():
            param.requires_grad = False
        for param in self.scale_val.parameters():
            param.requires_grad = False

    def unfreeze_val(self):
        for param in self.txt_val.parameters():
            param.requires_grad = True
        for param in self.scale_val.parameters():
            param.requires_grad = True

    def encode(self, embs):
        return self.encoder(embs.permute(0, 3, 1, 2))

    def decode(self, latents):
        return self.decoder(latents).permute(0, 2, 3, 1)

    def detect(self, embs):
        return self.detector(embs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, multilabels, manuals, ground_truths, actions, lstm_states, shuffled_ids):

        channel_order = ['enemy', 'message', 'goal']

        movements = []
        roles = []
        for triplet in ground_truths:
            # sort entities by the order of the channels (enemy, message, goal)
            ordered_triplet = []
            for r in ROLE_ORDER:
                for e in triplet:
                    if e[2] == r:
                        ordered_triplet.append(e)
                        break
            movements.append([MOVEMENT_TYPES[e[1]] for e in ordered_triplet] + [3])
            roles.append([ROLE_TYPES[e[2]] for e in ordered_triplet] + [3])

        movements = torch.tensor(movements).to(multilabels.device)
        roles = torch.tensor(roles).to(multilabels.device)

        b, h, w, c = multilabels.shape
        movements = movements.view(b, 1, 1, c).repeat(1, h, w, 1)
        movements_embed = self._select(self.movement_embeddings, movements)
        roles = roles.view(b, 1, 1, c).repeat(1, h, w, 1)
        roles_embed = self._select(self.role_embeddings, roles)

        mask = multilabels.view(b, h, w, c, 1).repeat(1, 1, 1, 1, movements_embed.shape[-1])
        # b x h x w x c x embed_dim
        movements_embed = movements_embed * mask
        roles_embed = roles_embed * mask
        # b x h x w x (c x embed_dim)
        movements_embed = movements_embed.view(b, h, w, -1)
        roles_embed = roles_embed.view(b, h, w, -1)
        movements_embed = movements_embed.permute((0, 3, 1, 2))
        roles_embed = roles_embed.permute((0, 3, 1, 2))

        embeddings = movements_embed + roles_embed

        #embeddings = batched_convert_multilabel_to_emb(multilabels, manuals, ground_truths, self)

        """
        if self.shuffle_ids:
            embeddings[..., :len(self.relevant_cls_idxs)] = torch.gather(input=embeddings[..., :len(self.relevant_cls_idxs)], dim=-1, index=shuffled_ids.unsqueeze(1).unsqueeze(1).expand(-1, 10, 10, -1))
        if "sprite" in self.dropout_loc:
            F.dropout(embeddings[..., :len(self.relevant_cls_idxs)], p=self.dropout_prob, training=self.training, inplace=True)
        """

        latents = self.encoder(embeddings)
        latents = self.before_lstm_projector(latents)

        actions = self.action_embeddings(actions)

        mem_ins = torch.cat((latents, actions), dim=-1).unsqueeze(0)
        mem_outs, (hidden_states, cell_states) = self.lstm(mem_ins, lstm_states)
        mem_outs = mem_outs.squeeze(0)

        decoder_inps = self.after_lstm_projector(mem_outs)
        decoder_inps = decoder_inps.view(decoder_inps.shape[0], *self.after_encoder_shape[1:])

        pred_grid_logits = self.decoder(decoder_inps)
        pred_nonexistence_logits = self.nonexistence_head(mem_outs)

        """
        pred_latents = self.projection(mem_outs.squeeze(0))
        pred_nonexistence_logits = None
        if self.prediction_type == "location":
            pred_nonexistence_logits = self.nonexistence(pred_latents)
        pred_grid_logits = self.detect(self.decode(pred_latents))

        if self.shuffle_ids:
            pred_grid_logits[..., self.relevant_cls_idxs] = torch.gather(input=pred_grid_logits[..., self.relevant_cls_idxs], dim=-1, index=torch.argsort(shuffled_ids, dim=-1).unsqueeze(1).unsqueeze(1).expand(-1, 10, 10, -1))
            pred_nonexistence_logits[..., self.relevant_cls_idxs] = torch.gather(input=pred_nonexistence_logits[..., self.relevant_cls_idxs], dim=-1, index=torch.argsort(shuffled_ids, dim=-1))
        """

        pred_rewards = self.reward_head(mem_outs)
        pred_done_logits = self.done_head(mem_outs)

        return ((pred_grid_logits, pred_nonexistence_logits), pred_rewards, pred_done_logits), (hidden_states, cell_states)

    def grid_loss(self, grid_logits, nonexistence_logits, grid_probs):
        """
        if self.prediction_type == "existence":
            loss = F.binary_cross_entropy_with_logits(grid_logits, probs, pos_weight=self.pos_weight)
        elif self.prediction_type == "class":
            loss = F.cross_entropy(grid_logits.flatten(0, 2), probs.flatten(0, 2), weight=self.cls_weight)
        elif self.prediction_type == "location":
            all_logits = torch.cat((grid_logits.permute(0, 3, 1, 2).flatten(2, 3),
                                    nonexistence_logits.unsqueeze(-1)), dim=-1).flatten(0, 1)
            nonexistence_probs = 1.0*(torch.sum(probs, dim=(1, 2)) <= 0)
            all_probs = torch.cat((probs.permute(0, 3, 1, 2).flatten(2, 3),
                                   nonexistence_probs.unsqueeze(-1)), dim=-1).flatten(0, 1)
            loss = F.cross_entropy(all_logits, all_probs, weight=self.loc_weight)
        else:
            raise NotImplementedError
        """
        grid_logits = grid_logits.view(grid_logits.shape[0], grid_logits.shape[1], -1)
        nonexistence_logits = nonexistence_logits.unsqueeze(-1)
        location_logits = torch.cat((grid_logits, nonexistence_logits), dim=-1)

        grid_probs = grid_probs.permute((0, 3, 1, 2))
        grid_probs = grid_probs.view(grid_probs.shape[0], grid_probs.shape[1], -1)
        nonexistence_probs = (1. - grid_probs.sum(dim=-1)).unsqueeze(-1)
        location_probs = torch.cat((grid_probs, nonexistence_probs), dim=-1)

        loss = F.cross_entropy(location_logits, location_probs)
        #location_logits =

        return loss

    def reward_loss(self, pred_rewards, rewards):
        return F.mse_loss(pred_rewards, rewards)

    def done_loss(self, pred_done_logits, done_probs):
        return F.binary_cross_entropy_with_logits(pred_done_logits, done_probs)

    def logit_to_prob(self, grid_logits, nonexistence_logits):
        nonexistence_probs = None
        if self.prediction_type == "existence":
            probs = torch.sigmoid(grid_logits)
        elif self.prediction_type == "class":
            probs = F.softmax(grid_logits, dim=-1)
        elif self.prediction_type == "location":
            all_logits = torch.cat((grid_logits.permute(0, 3, 1, 2).flatten(2, 3), nonexistence_logits.unsqueeze(-1)), dim=-1)
            all_probs = F.softmax(all_logits, dim=-1)
            probs = all_probs[..., :-1].unflatten(-1, (10, 10)).permute(0, 2, 3, 1)
            nonexistence_probs = all_probs[..., -1]
        else:
            raise NotImplementedError
        return probs, nonexistence_probs

    def multilabel_to_prob(self, multilabels):
        if self.prediction_type == "existence":
            probs = multilabels.float()
        elif self.prediction_type == "class":
            probs = multilabels / multilabels.sum(dim=-1, keepdim=True)
        elif self.prediction_type == "location":
            probs = multilabels.float() # assumes only at most one entity per class exists
        else:
            raise NotImplementedError
        return probs

    # reset hidden states for real prediction
    def real_state_reset(self, init_grids, idxs=None):
        if idxs is None:
            self.real_hidden_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.real_cell_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.real_entity_ids = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values
            self.real_shuffled_ids = None
            """
            if self.shuffle_ids:
                self.real_shuffled_ids = torch.from_numpy(np.random.default_rng().permuted(np.broadcast_to(np.arange(len(self.relevant_cls_idxs)), (self.batch_size, len(self.relevant_cls_idxs))), axis=-1)).long().to(self.device)
            """
        else:
            init_grids = init_grids[idxs]
            self.real_hidden_states[:, idxs] = 0
            self.real_cell_states[:, idxs] = 0
            self.real_entity_ids[idxs] = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values
            """
            if self.shuffle_ids:
                self.real_shuffled_ids[idxs] = torch.from_numpy(np.random.default_rng().permuted(np.broadcast_to(np.arange(len(self.relevant_cls_idxs)), (len(idxs), len(self.relevant_cls_idxs))), axis=-1)).long().to(self.device)
            """


    # reset hidden states for imag prediction
    def imag_state_reset(self, init_grids, idxs=None):
        if idxs is None:
            self.imag_hidden_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.imag_cell_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.imag_old_multilabels = batched_convert_grid_to_multilabel(init_grids)
            self.imag_entity_ids = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values
            self.imag_shuffled_ids = None
            if self.shuffle_ids:
                self.imag_shuffled_ids = torch.from_numpy(np.random.default_rng().permuted(np.broadcast_to(np.arange(len(self.relevant_cls_idxs)), (self.batch_size, len(self.relevant_cls_idxs))), axis=-1)).long().to(self.device)
        else:
            init_grids = init_grids[idxs]
            self.imag_hidden_states[:, idxs] = 0
            self.imag_cell_states[:, idxs] = 0
            self.imag_old_multilabels[idxs] = batched_convert_grid_to_multilabel(init_grids)
            self.imag_entity_ids[idxs] = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values
            if self.shuffle_ids:
                self.imag_shuffled_ids[idxs] = torch.from_numpy(np.random.default_rng().permuted(np.broadcast_to(np.arange(len(self.relevant_cls_idxs)), (len(idxs), len(self.relevant_cls_idxs))), axis=-1)).long().to(self.device)


    # detach hidden states for real prediction
    def real_state_detach(self):
        self.real_hidden_states = self.real_hidden_states.detach()
        self.real_cell_states = self.real_cell_states.detach()

    # detach hidden states for imag prediction
    def imag_state_detach(self):
        self.imag_hidden_states = self.imag_hidden_states.detach()
        self.imag_cell_states = self.imag_cell_states.detach()
        self.imag_old_multilabels = self.imag_old_multilabels.detach()

    # make real prediction and accumulate real loss
    def real_step(self, old_grids, manuals, ground_truths, actions, grids, rewards, dones, backprop_idxs):
        #old_multilabels = batched_convert_grid_to_multilabel(old_grids)
        #multilabels = batched_convert_grid_to_multilabel(grids)
        old_multilabels = old_grids
        old_multilabels[old_multilabels > 0] = 1.
        multilabels = grids
        multilabels[multilabels > 0] = 1.
        #probs = self.multilabel_to_prob(multilabels)
        done_probs = dones.float()

        (pred_loc_logits, pred_rewards, pred_done_logits), (self.real_hidden_states, self.real_cell_states) = \
            self.forward(
                    old_multilabels,
                    manuals,
                    ground_truths,
                    actions,
                    (self.real_hidden_states, self.real_cell_states),
                    self.real_shuffled_ids)

        pred_grid_logits, pred_nonexistence_logits = pred_loc_logits
        #n_backprops = len(backprop_idxs)
        self.real_grid_loss_total += self.grid_loss(
            pred_grid_logits[backprop_idxs],
            pred_nonexistence_logits[backprop_idxs],
            multilabels[backprop_idxs]
        )
        self.real_reward_loss_total += self.reward_loss(
            pred_rewards[backprop_idxs], rewards[backprop_idxs])
        self.real_done_loss_total += self.done_loss(
            pred_done_logits[backprop_idxs], done_probs[backprop_idxs])
        #self.real_backprop_count += n_backprops

        """
        with torch.no_grad():
            pred_grid_probs, pred_nonexistence_probs = self.logit_to_prob(pred_grid_logits, pred_nonexistence_logits)
            pred_multilabels = batched_convert_prob_to_multilabel(pred_grid_probs, pred_nonexistence_probs, self.prediction_type, self.pred_multilabel_threshold, self.refine_pred_multilabel, self.real_entity_ids)
            pred_done_probs = torch.sigmoid(pred_done_logits)
        return (((pred_grid_probs, pred_nonexistence_probs), pred_multilabels), pred_rewards, pred_done_probs), ((probs, multilabels), rewards, done_probs)
        """

    """
    # make imag prediction and accumulate imag loss
    def imag_step(self, manuals, ground_truths, actions, grids, rewards, dones, backprop_idxs):
        old_multilabels = self.imag_old_multilabels
        multilabels = batched_convert_grid_to_multilabel(grids)
        probs = self.multilabel_to_prob(multilabels)
        done_probs = dones.float()

        (pred_loc_logits, pred_rewards, pred_done_logits), (self.imag_hidden_states, self.imag_cell_states) = self.forward(old_multilabels, manuals, ground_truths, actions, (self.imag_hidden_states, self.imag_cell_states), self.imag_shuffled_ids)
        pred_grid_logits, pred_nonexistence_logits = pred_loc_logits
        n_backprops = len(backprop_idxs)
        self.imag_grid_loss_total += n_backprops*self.grid_loss(pred_grid_logits[backprop_idxs][..., self.relevant_cls_idxs], pred_nonexistence_logits[backprop_idxs][..., self.relevant_cls_idxs], probs[backprop_idxs][..., self.relevant_cls_idxs])
        self.imag_reward_loss_total += n_backprops*self.reward_loss(pred_rewards[backprop_idxs], rewards[backprop_idxs])
        self.imag_done_loss_total += n_backprops*self.done_loss(pred_done_logits[backprop_idxs], done_probs[backprop_idxs])
        self.imag_backprop_count += n_backprops

        with torch.no_grad():
            pred_grid_probs, pred_nonexistence_probs = self.logit_to_prob(pred_grid_logits, pred_nonexistence_logits)
            pred_multilabels = batched_convert_prob_to_multilabel(pred_grid_probs, pred_nonexistence_probs, self.prediction_type, self.pred_multilabel_threshold, self.refine_pred_multilabel, self.imag_entity_ids)
            self.imag_old_multilabels = pred_multilabels
            pred_done_probs = torch.sigmoid(pred_done_logits)
        return (((pred_grid_probs, pred_nonexistence_probs), pred_multilabels), pred_rewards, pred_done_probs), ((probs, multilabels), rewards, done_probs)
    """

    # update model via real loss
    def real_loss_update(self):
        self.optimizer.zero_grad()
        real_loss_mean = self.real_grid_loss_total + \
            self.reward_loss_weight * self.real_reward_loss_total + \
            self.done_loss_weight * self.real_done_loss_total
        print(real_loss_mean.item())
        real_loss_mean.backward()
        self.optimizer.step()

    """
    # update model via imag loss
    def imag_loss_update(self):
        self.optimizer.zero_grad()
        imag_loss_mean = (self.imag_grid_loss_total + self.reward_loss_weight*self.imag_reward_loss_total + self.done_loss_weight*self.imag_done_loss_total) / self.imag_backprop_count
        imag_loss_mean.backward()
        self.optimizer.step()
    """

    # reset real loss
    def real_loss_reset(self):
        with torch.no_grad():
            real_grid_loss_mean = self.real_grid_loss_total / self.real_backprop_count
            real_reward_loss_mean = self.real_reward_loss_total / self.real_backprop_count
            real_done_loss_mean = self.real_done_loss_total / self.real_backprop_count
            real_loss_mean = (self.real_grid_loss_total + self.reward_loss_weight*self.real_reward_loss_total + self.done_loss_weight*self.real_done_loss_total) / self.real_backprop_count
        self.real_grid_loss_total = 0
        self.real_reward_loss_total = 0
        self.real_done_loss_total = 0
        self.real_backprop_count = 0
        return real_grid_loss_mean.item(), real_reward_loss_mean.item(), real_done_loss_mean.item(), real_loss_mean.item()

    """
    # reset imag loss
    def imag_loss_reset(self):
        with torch.no_grad():
            imag_grid_loss_mean = self.imag_grid_loss_total / self.imag_backprop_count
            imag_reward_loss_mean = self.imag_reward_loss_total / self.imag_backprop_count
            imag_done_loss_mean = self.imag_done_loss_total / self.imag_backprop_count
            imag_loss_mean = (self.imag_grid_loss_total + self.reward_loss_weight*self.imag_reward_loss_total + self.done_loss_weight*self.imag_done_loss_total) / self.imag_backprop_count
        self.imag_grid_loss_total = 0
        self.imag_reward_loss_total = 0
        self.imag_done_loss_total = 0
        self.imag_backprop_count = 0
        return imag_grid_loss_mean.item(), imag_reward_loss_mean.item(), imag_done_loss_mean.item(), imag_loss_mean.item()
    """
