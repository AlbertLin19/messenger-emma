import math
from copy import deepcopy
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoConfig, AutoModel

from offline_training.batched_world_model.modules import ResNetEncoder, ResNetDecoder, ScaledDotProductAttention
from offline_training.batched_world_model.utils import batched_convert_grid_to_multilabel, batched_convert_multilabel_to_emb, batched_convert_prob_to_multilabel

from messenger.envs.config import NPCS, NO_MESSAGE, WITH_MESSAGE

ENTITY_IDS = {entity.name: entity.id for entity in NPCS}
MOVEMENT_TYPES = {
    "chaser": 0,
    "fleeing": 1,
    "immovable": 2,
    "unknown": 5,
}
ROLE_TYPES = {
    "message": 1,
    "goal": 2,
    "enemy": 0,
    "unknown": 5,
}

ROLE_ORDER = ['enemy', 'message', 'goal']

H = 10
W = 10
GRID_CHANNELS = 4
NUM_ENTITIES = 17


class WorldModelBase(nn.Module):

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def loc_loss(self, logits, targets):
        logits = logits.view(-1, logits.shape[-1])
        targets  = targets.view(-1, targets.shape[-1])
        loss = F.cross_entropy(logits, targets, reduction='sum') / (self.batch_size * GRID_CHANNELS)
        return loss

    def id_loss(self, logits, targets):

        if self.manuals in ['gpt', 'oracle']:
            normalizer = self.batch_size
        else:
            normalizer = self.batch_size * GRID_CHANNELS

        #normalizer = self.batch_size
        #print(logits[0].max(-1)[1].tolist(), targets[0].tolist())

        logits = logits.view(-1, logits.shape[-1])
        targets = targets.view(-1)

        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=-1,
            reduction='sum'
        )

        loss /= normalizer

        return loss

    def reward_loss(self, preds, targets):
        return F.mse_loss(preds, targets, reduction='sum') / self.batch_size

    def done_loss(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='sum') / self.batch_size

    # reset hidden states for real prediction
    def state_reset(self, init_grids, idxs=None):
        if idxs is None:
            self.hidden_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.cell_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
        else:
            self.hidden_states[:, idxs] = 0
            self.cell_states[:, idxs] = 0

    # detach hidden states for real prediction
    def state_detach(self):
        self.hidden_states = self.hidden_states.detach()
        self.cell_states = self.cell_states.detach()

    # update model via real loss
    def loss_update(self):
        self.optimizer.zero_grad()

        if self.train_loc_loss_only:
            total_loss = self.loss['loc']
        elif self.train_id_loss_only:
            total_loss = self.loss['id']
        elif self.train_reward_and_done_losses_only:
            total_loss = self.loss['reward'] + self.loss['done']
        elif self.train_all_losses_except_loc:
            total_loss = 0.
            for k in self.loss:
                if k == 'loc':
                    continue
                total_loss += self.loss[k] * self.loss_weight[k]
        else:
            total_loss = 0.
            for k in self.loss:
                total_loss += self.loss[k] * self.loss_weight[k]

        total_loss.backward()
        self.optimizer.step()

        #print(total_loss.item() / self.backprop_count)

        loss_values = self.loss_reset()
        self.state_detach()

        return loss_values

    # reset real loss
    def loss_reset(self):
        with torch.no_grad():
            avg_loss = {}
            avg_loss['total'] = 0
            for k in self.loss:
                avg_loss[k] = self.loss[k].item() / self.backprop_count
                avg_loss['total'] += avg_loss[k] * self.loss_weight[k]

        for k in self.loss:
            self.loss[k] = 0

        self.backprop_count = 0

        return avg_loss

    def predict_grid(self, preds, sample=False):

        grids = torch.zeros((preds['loc'].shape[0], H, W, GRID_CHANNELS)).to(self.device)
        if sample:
            loc_dists = torch.distributions.Categorical(probs=preds['loc'])
            locs = loc_dists.sample()
            id_dists = torch.distributions.Categorical(probs=preds['id'])
            ids = id_dists.sample()
        else:
            _, locs = preds['loc'].max(-1)
            _, ids = preds['id'].max(-1)

        grids = F.one_hot(locs, num_classes=H * W + 1)
        grids = grids[...,:-1]
        grids = grids.view(grids.shape[0], grids.shape[1], H,  W)

        ids = ids.view(ids.shape[0], ids.shape[1], 1, 1).repeat(1, 1, H, W)
        grids = grids * ids
        # b x c x h x w --> b x h x w x c
        grids = grids.permute((0, 2, 3, 1))

        return grids


class WorldModel(WorldModelBase):

    def __init__(self, args):

        super().__init__()

        self.batch_size = args.batch_size
        self.device = args.device
        self.manuals = args.manuals
        self.keep_entity_features_for_parsed_manuals = args.keep_entity_features_for_parsed_manuals
        self.use_true_attention = args.use_true_attention
        self.train_id_loss_only = args.train_id_loss_only
        self.train_loc_loss_only = args.train_loc_loss_only
        self.train_reward_and_done_losses_only = args.train_reward_and_done_losses_only
        self.train_all_losses_except_loc = args.train_all_losses_except_loc

        self.hidden_size = hidden_size = args.hidden_size
        self.attr_embed_dim = attr_embed_dim = args.attr_embed_dim
        self.desc_key_dim = desc_key_dim = args.desc_key_dim
        action_embed_dim = args.action_embed_dim

        self.pos_embeddings = nn.Embedding(GRID_CHANNELS, attr_embed_dim)
        self.id_embeddings = nn.Embedding(NUM_ENTITIES, attr_embed_dim, padding_idx=0)
        self.role_embeddings = nn.Embedding(len(ROLE_TYPES) + 3, attr_embed_dim)
        self.movement_embeddings = nn.Embedding(len(MOVEMENT_TYPES) + 3, attr_embed_dim)

        # manual attention
        text_embed_dim = 768
        self.manual_key_scale = nn.Linear(text_embed_dim, 1)
        self.manual_key_linear = nn.Linear(text_embed_dim, attr_embed_dim)
        self.manual_value_scale = nn.Linear(text_embed_dim, 1)
        self.manual_value_linear = nn.Linear(text_embed_dim, attr_embed_dim)
        self.manual_attention = ScaledDotProductAttention(attr_embed_dim)

        """
        bert_config = AutoConfig.from_pretrained(
            'prajjwal1/bert-tiny',
            hidden_size=text_embed_dim,
            vocab_size=1,
            num_attention_heads=1,
            num_hidden_layers=1,
            max_position_embeddings=40,
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0
        )
        self.manual_key_linear = nn.Linear(text_embed_dim, attr_embed_dim)
        self.manual_value_linear = nn.Linear(text_embed_dim, attr_embed_dim)
        self.manual_key_transformer = AutoModel.from_config(bert_config)
        self.manual_value_transformer = AutoModel.from_config(bert_config)
        self.manual_attention = ScaledDotProductAttention(attr_embed_dim)
        """


        """
        self.movement_value_proj = nn.Linear(text_embed_dim, attr_embed_dim)
        self.movement_key_proj = nn.Linear(text_embed_dim, attr_embed_dim)
        self.movement_attention = ScaledDotProductAttention(attr_embed_dim)

        self.role_value_proj = nn.Linear(text_embed_dim, attr_embed_dim)
        self.role_key_proj = nn.Linear(text_embed_dim, attr_embed_dim)
        self.role_attention = ScaledDotProductAttention(attr_embed_dim)
        """

        """
        self.entity_query_embeddings = nn.Embedding(NUM_ENTITIES, desc_key_dim)
        self.token_key = nn.Linear(768, desc_key_dim)
        self.token_key_att = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Softmax(dim=-2)
        )
        self.token_movement_val = nn.Linear(768, attr_embed_dim)
        self.token_movement_val_att = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Softmax(dim=-2)
        )
        self.token_role_val = nn.Linear(768, attr_embed_dim)
        self.token_role_val_att = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Softmax(dim=-2)
        )
        """

        self.encoder = ResNetEncoder(attr_embed_dim)
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

        self.location_head = ResNetDecoder(attr_embed_dim, GRID_CHANNELS)

        self.id_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, GRID_CHANNELS * NUM_ENTITIES)
        )

        nonexistence_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ]
        for _ in range(self.encoder.num_layers - 1):
            nonexistence_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])
        nonexistence_layers.append(nn.Linear(hidden_size, GRID_CHANNELS))
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
        self.loss = {
            'loc': 0,
            'id': 0,
            'reward': 0,
            'done': 0
        }
        self.loss_weight = args.loss_weights
        self.id_class_count = [1] * NUM_ENTITIES
        self.backprop_count = 0

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def embed_grids_with_parsed_manuals(self, grids, parsed_manuals):
        positions = [] # B x 4
        movements = [] # B x 4
        roles = [] # B x 4
        for triplet in parsed_manuals:
            movements.append([MOVEMENT_TYPES[e[1]] if e is not None else 4 for e in triplet] + [3])
            roles.append([ROLE_TYPES[e[2]] if e is not None else 4 for e in triplet] + [3])
            positions.append([0, 1, 2, 3])

        movements = torch.tensor(movements).to(self.device)
        roles = torch.tensor(roles).to(self.device)
        positions = torch.tensor(positions).to(self.device)

        # movements, roles, positions are B x 4 tensors of indices, where [b, i] is the index of m/r/p for the ith channel

        b, h, w, c = grids.shape
        movements = movements.view(b, 1, 1, c).repeat(1, h, w, 1) # B x H x W x 4
        movements_embed = self._select(self.movement_embeddings, movements)

        roles = roles.view(b, 1, 1, c).repeat(1, h, w, 1)
        roles_embed = self._select(self.role_embeddings, roles)

        positions = positions.view(b, 1, 1, c).repeat(1, h, w, 1)
        positions_embed = self._select(self.pos_embeddings, positions)

        # m/r/p embed are B x h x w x 4 x embed_dim, where [b, h, w, i] is the embed of ith channel (and h, w is to broadcast against grid)

        mask = (grids > 0).float().unsqueeze(-1)
        # b x h x w x c x embed_dim
        movements_embed = movements_embed * mask
        roles_embed = roles_embed * mask
        positions_embed = positions_embed * mask

        # now, embeds have nonzero only in correct grid position

        # b x h x w x embed_dim
        movements_embed = movements_embed.sum(dim=-2)
        roles_embed = roles_embed.sum(dim=-2)
        positions_embed = positions_embed.sum(dim=-2)

        # dimension of size 4 is collapsed

        # b x embed_dim x h x w
        movements_embed = movements_embed.permute((0, 3, 1, 2))
        roles_embed = roles_embed.permute((0, 3, 1, 2))
        positions_embed = positions_embed.permute((0, 3, 1, 2))

        grid_ids = grids.clone().long()
        # zero out entity ids in the first 3 channels
        if not self.keep_entity_features_for_parsed_manuals:
            grid_ids[..., :-1][grid_ids[..., :-1] > 0] = 0
        # b x h x w x c x embed_dim
        ids_embed = self._select(self.id_embeddings, grid_ids)
        # b x embed_dim x h x w
        ids_embed = ids_embed.sum(dim=-2)
        # b x embed_dim x h x w
        ids_embed = ids_embed.permute((0, 3, 1, 2))

        return roles_embed + movements_embed + positions_embed + ids_embed
        #return positions_embed + ids_embed

    def embed_grids_with_embedded_manuals(self, grids, embedded_manuals, manuals_attn_mask):
        # convert grids: b x h x w x c and embedded_manuals: b x 3 x 36 (sent_len) x 768 (emb_dim)
        # to embed_grids: b x embed_dim x h x w

        b, h, w, c = grids.shape
        positions = [[0, 1, 2, 3] for _ in embedded_manuals] # B x 4
        positions = torch.tensor(positions).to(self.device)
        positions = positions.view(b, 1, 1, c).repeat(1, h, w, 1)
        positions_embed = self._select(self.pos_embeddings, positions)

        # create an empty manual
        #embedded_manuals = embedded_manuals.mean(dim=-2)
        """
        _, _, l, d = embedded_manuals.shape
        empty_manual = torch.zeros((b, 1, l, d)).to(self.device)
        embedded_manuals = torch.cat((embedded_manuals, empty_manual), dim=1)
        """


        """
        flat_embedded_manuals = embedded_manuals.view(-1, l, d)

        manuals_key = self.manual_key_transformer.forward(inputs_embeds=flat_embedded_manuals).last_hidden_state
        manuals_key = manuals_key.view(b, c, l, d)
        manuals_key = manuals_key.mean(dim=-2)
        manuals_key = self.manual_key_linear(manuals_key)

        manuals_value = self.manual_value_transformer.forward(inputs_embeds=flat_embedded_manuals).last_hidden_state
        manuals_value = manuals_value.view(b, c, l, d)
        manuals_value = manuals_value.mean(dim=-2)
        manuals_value = self.manual_value_linear(manuals_value)
        """

        # key
        manuals_key = self.manual_key_linear(embedded_manuals)
        manuals_key_scale = self.manual_key_scale(embedded_manuals)
        manuals_key_scale.masked_fill_(manuals_attn_mask.unsqueeze(-1), -float('inf'))
        manuals_key_scale = manuals_key_scale.softmax(dim=-2)
        manuals_key = (manuals_key * manuals_key_scale).sum(dim=-2)

        # value
        manuals_value = self.manual_value_linear(embedded_manuals)
        manuals_value_scale = self.manual_value_scale(embedded_manuals)
        manuals_value_scale.masked_fill_(manuals_attn_mask.unsqueeze(-1), -float('inf'))
        manuals_value_scale = manuals_value_scale.softmax(dim=-2)
        manuals_value = (manuals_value * manuals_value_scale).sum(dim=-2)

        # query
        entities, _ = grids.view(b, -1, c).max(1)
        entities_query = self._select(self.id_embeddings, entities)

        true_attn = self.true_attn if self.use_true_attention else None
        attr_embed, self.attn = self.manual_attention(
            entities_query[:,:-1,:],  # exclude the player
            manuals_key,
            manuals_value,
            true_attn=true_attn
        )
        player_embed = entities_query[:,-1,:]
        attr_embed = torch.cat((attr_embed, player_embed.unsqueeze(1)), dim=1)
        attr_embed = attr_embed.view(b, 1, 1, c, attr_embed.shape[-1]).repeat(1, h, w, 1, 1)

        #print(entities[0].tolist())
        #print(attn[0, 0])

        """
        # movement attention
        movements_key = self.movement_key_proj(embedded_manuals)
        movements_value = self.movement_value_proj(embedded_manuals)
        movements_embed, _ = self.movement_attention(
            entities_query,
            movements_key,
            movements_value
        )
        movements_embed = movements_embed.view(b, 1, 1, c, movements_embed.shape[-1]).repeat(1, h, w, 1, 1)

        # role attention
        roles_key = self.role_key_proj(embedded_manuals)
        roles_value = self.role_value_proj(embedded_manuals)
        roles_embed, _ = self.role_attention(
            entities_query,
            roles_key,
            roles_value
        )
        roles_embed = roles_embed.view(b, 1, 1, c, roles_embed.shape[-1]).repeat(1, h, w, 1, 1)
        """

        #print(movements_embed.shape, roles_embed.shape)

        """
        # need to construct movements_embed, roles_embed which are b x h x w x 4 x embed_dim, where [b, h, w, i] is the embed of ith channel
        # compute attentions over descriptions
        entity_query_embed = self._select(self.entity_query_embeddings, grids) # b x h x w x c x desc_key_dim
        desc_key = torch.sum(self.token_key_att(embedded_manuals)*self.token_key(embedded_manuals), dim=-2) # b x 3 (num_sent) x desc_key_dim
        desc_att_logits = torch.matmul(entity_query_embed, desc_key.permute(0, 2, 1).view(b, 1, 1, self.desc_key_dim, 3)) # b x h x w x c x 3 (num_sent)
        desc_att = F.softmax(desc_att_logits / np.sqrt(self.desc_key_dim), dim=-1)
        # compute description movement/role values
        desc_movement_val = torch.sum(self.token_movement_val_att(embedded_manuals)*self.token_movement_val(embedded_manuals), dim=-2) # b x 3 (num_sent) x attr_embed_dim
        desc_role_val = torch.sum(self.token_role_val_att(embedded_manuals)*self.token_role_val(embedded_manuals), dim=-2) # b x 3 (num_sent) x attr_embed_dim
        # compute movements_embed and roles_embed
        movements_embed = torch.sum(desc_att.view(b, h, w, c, 3, 1)*desc_movement_val.view(b, 1, 1, 1, 3, self.attr_embed_dim), dim=-2) # b x h x w x c x attr_embed_dim
        roles_embed = torch.sum(desc_att.view(b, h, w, c, 3, 1)*desc_role_val.view(b, 1, 1, 1, 3, self.attr_embed_dim), dim=-2) # b x h x w x c x attr_embed_dim
        # at this point, movements_embed and roles_embed are a batch of grids with channel depth 4, where each entry is a value encoding representative of the id at that location in the original grid
        # need to zero-out the entries where ID is 0 (done later), and need to set avatar channel to be its own custom embedding
        movements_embed[..., -1, :] = self.movement_embeddings(torch.tensor([3], device=self.device))
        roles_embed[..., -1, :] = self.role_embeddings(torch.tensor([3], device=self.device))
        """

        mask = (grids > 0).float().unsqueeze(-1)
        # b x h x w x c x embed_dim
        """
        movements_embed = movements_embed * mask
        roles_embed = roles_embed * mask
        """
        attr_embed = attr_embed * mask
        positions_embed = positions_embed * mask

        # now, embeds have nonzero only in correct grid position

        # b x h x w x embed_dim
        """
        movements_embed = movements_embed.sum(dim=-2)
        roles_embed = roles_embed.sum(dim=-2)
        """
        attr_embed = attr_embed.sum(dim=-2)
        positions_embed = positions_embed.sum(dim=-2)

        # dimension of size 4 is collapsed

        # b x embed_dim x h x w
        """
        movements_embed = movements_embed.permute((0, 3, 1, 2))
        roles_embed = roles_embed.permute((0, 3, 1, 2))
        """
        attr_embed = attr_embed.permute((0, 3, 1, 2))
        positions_embed = positions_embed.permute((0, 3, 1, 2))

        grid_ids = grids.clone().long()
        #print('===>', grid_ids.view(b, -1, c)[0].max(0)[0].tolist())
        # b x h x w x c x embed_dim
        ids_embed = self._select(self.id_embeddings, grid_ids)
        # b x embed_dim x h x w
        ids_embed = ids_embed.sum(dim=-2)
        # b x embed_dim x h x w
        ids_embed = ids_embed.permute((0, 3, 1, 2))

        #return roles_embed + movements_embed + positions_embed + ids_embed
        return attr_embed + positions_embed #+ ids_embed
        #return positions_embed + ids_embed

    def embed_grids_without_manuals(self, grids):

        b, h, w, c = grids.shape
        positions = torch.arange(0, 4).to(self.device)
        positions = positions.view(1, 1, 1, c).repeat(b, h, w, 1)
        positions_embed = self._select(self.pos_embeddings, positions)
        mask = (grids > 0).float().unsqueeze(-1)
        positions_embed = positions_embed * mask
        positions_embed = positions_embed.sum(dim=-2)
        positions_embed = positions_embed.permute((0, 3, 1, 2))

        ids_embed = self._select(self.id_embeddings, grids.long())
        ids_embed = ids_embed.sum(dim=-2)
        ids_embed = ids_embed.permute((0, 3, 1, 2))

        return positions_embed + ids_embed


    def forward(self, grids, embedded_manuals, manuals_attn_mask, parsed_manuals, actions, lstm_states):

        if self.manuals in ['gpt', 'oracle']:
            grids_embed = self.embed_grids_with_parsed_manuals(grids, parsed_manuals)
        elif self.manuals == 'embed':
            grids_embed = self.embed_grids_with_embedded_manuals(grids, embedded_manuals, manuals_attn_mask)
        elif self.manuals == 'none':
            grids_embed = self.embed_grids_without_manuals(grids)
        else:
            raise NotImplementedError

        latents = self.encoder(grids_embed)
        latents = self.before_lstm_projector(latents)

        actions = self.action_embeddings(actions)

        mem_ins = torch.cat((latents, actions), dim=-1).unsqueeze(0)
        mem_outs, (hidden_states, cell_states) = self.lstm(mem_ins, lstm_states)
        mem_outs = mem_outs.squeeze(0)

        logits= {}

        loc_inps = self.after_lstm_projector(mem_outs)
        loc_inps = loc_inps.view(loc_inps.shape[0], *self.after_encoder_shape[1:])
        logits['grid'] = self.location_head(loc_inps)
        logits['nonexistence'] = self.nonexistence_head(mem_outs)

        logits['id'] = self.id_head(mem_outs)
        logits['id'] = logits['id'].view(logits['id'].shape[0], GRID_CHANNELS, NUM_ENTITIES)

        logits['reward'] = self.reward_head(mem_outs)
        logits['done'] = self.done_head(mem_outs)

        return logits, (hidden_states, cell_states)

    def create_loc_logits_and_targets(self, grid_logits, nonexistence_logits, grids):

        grid_logits = grid_logits.view(grid_logits.shape[0], grid_logits.shape[1], -1)
        nonexistence_logits = nonexistence_logits.unsqueeze(-1)
        location_logits = torch.cat((grid_logits, nonexistence_logits), dim=-1)

        grid_targets = (grids.permute((0, 3, 1, 2)) > 0).float()
        grid_targets = grid_targets.view(grid_targets.shape[0], grid_targets.shape[1], -1)
        nonexistence_targets = (1. - grid_targets.sum(dim=-1)).unsqueeze(-1)
        location_targets = torch.cat((grid_targets, nonexistence_targets), dim=-1)

        return location_logits, location_targets

    def create_id_logits_and_targets(self, id_logits, grids, true_parsed_manuals):
        id_targets = torch.zeros((id_logits.shape[0], GRID_CHANNELS), device=self.device).long()
        for i, triplet in enumerate(true_parsed_manuals):
            for j, e in enumerate(triplet):
                id_targets[i, j] = ENTITY_IDS[e[0]] if e is not None else 0
            avatar_id = grids[i, :, :, 3].view(-1).max().item()
            id_targets[i, 3] = avatar_id

        if self.manuals in ['gpt', 'oracle']:
            # only predict id of avatar
            id_targets[:, :3] = -1

        return id_logits, id_targets

    def reorder_parsed_manuals(self, parsed_manuals, grids):
        b, h, w, c = grids.shape
        entity_per_channel, _ = grids[...,:-1].flatten(1, 2).max(1)
        new_parsed_manuals = []
        for i, triplet in enumerate(parsed_manuals):
            new_triplet = []
            for r in entity_per_channel[i].tolist():
                if r == 0:
                    new_triplet.append(None)
                    continue
                new_triplet.append(("unknown", "unknown", "unknown")) # unknown by default
                for e in triplet:
                    if e[0] in ENTITY_IDS and ENTITY_IDS[e[0]] == r:
                        new_triplet[-1] = e
                        break
            assert len(new_triplet) == 3
            new_parsed_manuals.append(new_triplet)
        return new_parsed_manuals

    def make_true_attention(self, grids, true_parsed_manuals):
        b, h, w, c = grids.shape
        self.true_attn = torch.zeros((b, c - 1, c)).to(self.device)
        entities_batch = grids.view(b, -1, c).max(1)[0].tolist()
        for k, entities in enumerate(entities_batch):
            for i, e_grid in enumerate(entities[:-1]):
                if e_grid in [0, 15, 16]:
                    self.true_attn[k, i, -1] = 1
                else:
                    pos = -1
                    for j, e_manual in enumerate(true_parsed_manuals[k]):
                        if ENTITY_IDS[e_manual[0]] == e_grid:
                            pos = j
                            break
                    #assert pos is not None
                    self.true_attn[k, i, pos] = 1
        """
        print(ENTITY_IDS)
        print(true_parsed_manuals[5])
        print(entities_batch[5])
        print(self.true_attn[5])
        print()
        """

    def step(self,
            old_grids,
            embedded_manuals,
            manuals_attn_mask,
            parsed_manuals,
            true_parsed_manuals,
            actions,
            grids,
            rewards,
            dones,
            backprop_idxs):

        self.make_true_attention(old_grids, true_parsed_manuals)
        orig_true_parsed_manuals = deepcopy([true_parsed_manuals[i] for i in backprop_idxs])

        if parsed_manuals is not None:
            parsed_manuals = self.reorder_parsed_manuals(parsed_manuals, grids)
        true_parsed_manuals = self.reorder_parsed_manuals(true_parsed_manuals, grids)

        logits, (self.hidden_states, self.cell_states) = self.forward(
            old_grids,
            embedded_manuals,
            manuals_attn_mask,
            parsed_manuals,
            actions,
            (self.hidden_states, self.cell_states),
        )

        # select examples that need backprop
        if parsed_manuals is not None:
            parsed_manuals = [parsed_manuals[i] for i in backprop_idxs]
        true_parsed_manuals = [true_parsed_manuals[i] for i in backprop_idxs]
        for k in logits:
            logits[k] = logits[k][backprop_idxs]

        targets = {}
        targets['grid'] = grids[backprop_idxs]
        logits['loc'], targets['loc'] = self.create_loc_logits_and_targets(
            logits['grid'],
            logits['nonexistence'],
            targets['grid']
        )
        logits['id'], targets['id'] = self.create_id_logits_and_targets(
            logits['id'],
            grids,
            true_parsed_manuals
        )
        targets['reward'] = rewards[backprop_idxs]
        targets['done'] = dones[backprop_idxs].float()

        for k in targets:
            if k == 'grid':
                continue
            self.loss[k] += getattr(self, k + '_loss')(logits[k], targets[k])

        self.backprop_count += 1

        with torch.no_grad():
            preds = {}
            preds['loc'] = logits['loc'].softmax(dim=-1)
            preds['reward'] = logits['reward'].detach()
            preds['done'] = torch.sigmoid(logits['done'])
            preds['id'] = logits['id'].softmax(dim=-1)
            # oracle or gpt manuals: use parsed_manuals to overwrite id predictions
            if parsed_manuals is not None:
                for i, triplet in enumerate(parsed_manuals):
                    for j, e in enumerate(triplet):
                        preds['id'][i, j] = 0
                        if e is None:
                            k = 0
                        elif e[0] not in ENTITY_IDS:
                            k = 1
                        else:
                            k = ENTITY_IDS[e[0]]
                        preds['id'][i, j, k] = 1.
                preds['grid'] = self.predict_grid(preds)

            # attention
            preds['attn'] = self.attn[backprop_idxs][0]
            preds['true_attn'] = torch.zeros_like(preds['attn'])
            entities = old_grids[backprop_idxs][0].view(-1, grids.shape[-1]).max(0)[0].tolist()
            preds['entities'] = entities
            for i, e_grid in enumerate(entities[:-1]):
                if e_grid in [0, 15, 16]:
                    preds['true_attn'][i, -1] = 1
                else:
                    pos = None
                    for j, e_manual in enumerate(orig_true_parsed_manuals[0]):
                        if ENTITY_IDS[e_manual[0]] == e_grid:
                            pos = j
                            break
                    assert pos is not None
                    preds['true_attn'][i, pos] = 1

            preds['attn'] = torch.cat((preds['attn'], preds['true_attn']), dim=0)

            """
            print(ENTITY_IDS)
            print(orig_true_parsed_manuals[0])
            print(entities)
            print(preds['attn'])
            print()
            """


        return preds, targets
