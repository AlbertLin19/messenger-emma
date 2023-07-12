import random
import math
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from offline_training.batched_world_model.modules import ResNetEncoder, ResNetDecoder, LSTMDecoder
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
DESCRIPTION_LENGTH = 14


class WorldModelBase(nn.Module):

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    # reset hidden states for real prediction
    def state_reset(self, init_grids, idxs=None):
        if idxs is None:
            self.mem_states = (
                torch.zeros((1, self.batch_size, self.hidden_dim), device=self.device),
                torch.zeros((1, self.batch_size, self.hidden_dim), device=self.device)
            )
        else:
            self.mem_states[0][:, idxs] = 0
            self.mem_states[1][:, idxs] = 0

    def state_detach(self):
        self.mem_states = (
            self.mem_states[0].detach(),
            self.mem_states[1].detach()
        )

    def loss_update(self):

        self.optimizer.zero_grad()
        total_loss = 0
        for k in self.loss:
            if self.debug_latent_loss_only and k != 'latent':
                continue
            if self.debug_no_latent_loss and k == 'latent':
                continue
            total_loss += self.loss[k]

        total_loss.backward()
        self.optimizer.step()

        #print(self.loss['loc'].item() / self.backprop_count)

        loss_values = self.loss_reset()
        self.state_detach()

        return loss_values

    def loss_reset(self):

        with torch.no_grad():
            avg_loss = {}
            avg_loss['total'] = 0
            for k in self.loss:
                if self.debug_latent_loss_only and k != 'latent':
                    continue
                if self.debug_no_latent_loss and k == 'latent':
                    continue
                avg_loss[k] = self.loss[k].item() / self.backprop_count
                avg_loss['total'] += avg_loss[k]
                self.loss[k] = 0

        self.backprop_count = 0

        return avg_loss


class WorldModel(WorldModelBase):

    def __init__(self, args):

        super().__init__()

        self.random = random.Random(args.seed + 243)
        self.latent_tokenizer = args.latent_tokenizer
        self.batch_size = args.batch_size
        self.device = args.device
        self.manuals = args.manuals
        self.keep_entity_features_for_parsed_manuals = args.keep_entity_features_for_parsed_manuals
        self.debug_latent_loss_only = args.debug_latent_loss_only
        self.debug_zero_latent = args.debug_zero_latent
        self.debug_no_latent_loss = args.debug_no_latent_loss
        self.debug_no_reward_done_input = args.debug_no_reward_done_input
        self.debug_no_latent = args.debug_no_latent
        self.debug_no_predict_other_ids = args.debug_no_predict_other_ids

        if self.debug_no_latent:
            self.debug_no_latent_loss = 1

        self.hidden_dim = hidden_dim = args.hidden_dim
        self.attr_embed_dim = attr_embed_dim = args.attr_embed_dim
        self.desc_key_dim = desc_key_dim = args.desc_key_dim
        action_embed_dim = args.action_embed_dim

        self.pos_embeddings = nn.Embedding(GRID_CHANNELS, attr_embed_dim)
        self.id_embeddings = nn.Embedding(NUM_ENTITIES, attr_embed_dim, padding_idx=0)
        self.role_embeddings = nn.Embedding(len(ROLE_TYPES) + 3, attr_embed_dim)
        self.movement_embeddings = nn.Embedding(len(MOVEMENT_TYPES) + 3, attr_embed_dim)

        self.encoder = ResNetEncoder(attr_embed_dim)

        if args.manuals == "embed":
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

        test_in = torch.zeros((1, attr_embed_dim, 10, 10))
        test_out = self.encoder(test_in)
        self.after_encoder_shape = test_out.shape

        enc_dim = math.prod(self.after_encoder_shape[1:])
        self.before_mem_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU()
        )

        self.action_embeddings = nn.Embedding(5, action_embed_dim)
        self.reward_embeddings = nn.Embedding(5, action_embed_dim // 4)
        self.done_embeddings  = nn.Embedding(3, action_embed_dim // 4)

        mem_in_dim = hidden_dim + action_embed_dim + action_embed_dim // 2
        if self.debug_no_reward_done_input:
            mem_in_dim = hidden_dim + action_embed_dim
        self.memory = nn.LSTM(
            mem_in_dim,
            hidden_dim,
        )

        if not args.debug_no_latent:
            self.latent_input_projector = nn.Sequential(
                nn.Linear(hidden_dim, args.latent_dim),
                nn.ReLU()
            )
            self.latent_embeddings = nn.Embedding(
                len(args.latent_tokenizer.get_vocab()),
                args.latent_dim
            )
            self.latent_generator = LSTMDecoder(
                args.latent_tokenizer,
                args.latent_dim,
                args.device
            )
            # IMPORTANT: tie latent_generator embeddings with latent_embeddings
            self.latent_generator.embeddings.weight = self.latent_embeddings.weight

        loc_in_dim = hidden_dim
        if not args.debug_no_latent:
            loc_in_dim += args.latent_dim * 8
        self.loc_in_projector = nn.Sequential(
            nn.Linear(loc_in_dim, enc_dim),
            nn.ReLU()
        )
        self.location_head = ResNetDecoder(attr_embed_dim, GRID_CHANNELS)

        nonexistence_in_dim = hidden_dim
        if not args.debug_no_latent:
            nonexistence_in_dim += args.latent_dim * 8
        nonexistence_layers = [
            nn.Linear(nonexistence_in_dim, hidden_dim),
            nn.ReLU()
        ]
        for _ in range(self.encoder.num_layers - 1):
            nonexistence_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        nonexistence_layers.append(nn.Linear(hidden_dim, GRID_CHANNELS))
        self.nonexistence_head = nn.Sequential(*nonexistence_layers)

        id_in_dim = hidden_dim
        if not args.debug_no_latent:
            id_in_dim += args.latent_dim * 4
        self.id_head = nn.Sequential(
            nn.Linear(id_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, GRID_CHANNELS * NUM_ENTITIES)
        )

        reward_in_dim = hidden_dim
        if not args.debug_no_latent:
            reward_in_dim += args.latent_dim
        self.reward_head = nn.Sequential(
            nn.Linear(reward_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

        done_in_dim = hidden_dim
        if not args.debug_no_latent:
            done_in_dim += args.latent_dim
        self.done_head = nn.Sequential(
            nn.Linear(done_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        # training parameters
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.reward_loss_weight = args.reward_loss_weight
        self.done_loss_weight = args.done_loss_weight

        # loss accumulation
        self.loss = {
            'latent': 0,
            'loc' : 0,
            'id' : 0,
            'reward': 0,
            'done': 0
        }
        self.backprop_count = 0

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

    def embed_grids_with_embedded_manuals(self, grids, embedded_manuals):
        # convert grids: b x h x w x c and embedded_manuals: b x 3 x 36 (sent_len) x 768 (emb_dim)
        # to embed_grids: b x embed_dim x h x w

        positions = [[0, 1, 2, 3] for _ in embedded_manuals] # B x 4
        positions = torch.tensor(positions).to(self.device)
        b, h, w, c = grids.shape
        positions = positions.view(b, 1, 1, c).repeat(1, h, w, 1)
        positions_embed = self._select(self.pos_embeddings, positions)

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

    def step(self,
            manuals,
            grids,
            rewards,
            dones,
            actions,
            new_state_descriptions,
            mem_states,
            is_eval=False
        ):

        if self.manuals in ['gpt', 'oracle']:
            grids_embed = self.embed_grids_with_parsed_manuals(grids, manuals)
        elif self.manuals == 'embed':
            grids_embed = self.embed_grids_with_embedded_manuals(grids, manuals)
        elif self.manuals == 'none':
            grids_embed = self.embed_grids_without_manuals(grids)
        else:
            raise NotImplementedError

        grids_embed   = self.before_mem_projector(self.encoder(grids_embed))
        rewards_embed = self.reward_embeddings((rewards.long() + 1) * 2)
        dones_embed   = self.done_embeddings(dones)
        actions_embed = self.action_embeddings(actions)

        mem_ins = (grids_embed, rewards_embed, dones_embed, actions_embed)
        if self.debug_no_reward_done_input:
            mem_ins = (grids_embed, actions_embed)
        mem_ins = torch.cat(mem_ins, dim=-1).unsqueeze(0)
        mem_outs, new_mem_states = self.memory(mem_ins, mem_states)
        mem_outs = mem_outs.squeeze(0)

        logits = {}

        if not self.debug_no_latent:
            latent_init_states = self.latent_input_projector(mem_outs)
            logits['latent'] = self.latent_generator.decode(
                latent_init_states, new_state_descriptions[:, :-1])

            latents = new_state_descriptions[:, 1:]

            assert latents.shape[1] == DESCRIPTION_LENGTH

            latents = self.latent_embeddings(latents)

            if self.debug_zero_latent:
                latents = latents * 0

        if self.debug_no_latent:
            loc_ins = mem_outs
        else:
            loc_latents = latents[:, :8].flatten(1, 2)
            loc_ins = torch.cat([mem_outs, loc_latents], dim=-1)
        nonexistence_logits = self.nonexistence_head(loc_ins).unsqueeze(-1)
        loc_ins = self.loc_in_projector(loc_ins)
        loc_ins = loc_ins.view(loc_ins.shape[0], *self.after_encoder_shape[1:])
        grid_logits = self.location_head(loc_ins)
        grid_logits = grid_logits.view(grid_logits.shape[0], grid_logits.shape[1], -1)
        logits['loc'] = torch.cat((grid_logits, nonexistence_logits), dim=-1)

        if self.debug_no_latent:
            id_ins = mem_outs
        else:
            id_latents = latents[:, 8:12].flatten(1, 2)
            id_ins = torch.cat([mem_outs, id_latents], dim=-1)
        logits['id'] = self.id_head(id_ins)
        logits['id'] = logits['id'].view(logits['id'].shape[0], GRID_CHANNELS, NUM_ENTITIES)

        if self.debug_no_latent:
            reward_ins = mem_outs
        else:
            reward_latents = latents[:, 12]
            reward_ins = torch.cat([mem_outs, reward_latents], dim=-1)
        logits['reward'] = self.reward_head(reward_ins)

        if self.debug_no_latent:
            done_ins = mem_outs
        else:
            done_latents = latents[:, 13]
            done_ins = torch.cat([mem_outs, done_latents], dim=-1)
        logits['done'] = self.done_head(done_ins)

        """
        for k in logits:
            print(k, logits[k].shape)
        """

        return logits, new_mem_states

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

    def create_targets(self, grids, rewards, dones, state_descriptions, true_parsed_manuals):

        b, h, w, c = grids.shape

        targets = {}

        targets['latent'] = state_descriptions[:, 1:]
        assert targets['latent'].shape[1] == DESCRIPTION_LENGTH

        loc = (grids.permute((0, 3, 1, 2)) > 0).float()
        loc = loc.view(loc.shape[0], loc.shape[1], -1)
        nonexistence = (1. - loc.sum(dim=-1)).unsqueeze(-1)
        targets['loc'] = torch.cat((loc, nonexistence), dim=-1)
        assert targets['loc'].sum() == b * c
        targets['loc'] = targets['loc'].max(dim=-1)[1]

        targets['id'] = grids.view(b, -1, c).max(dim=1)[0]

        """
        targets['id'] = torch.zeros((b, c), device=self.device).long()
        for i, triplet in enumerate(true_parsed_manuals):
            for j, e in enumerate(triplet):
                targets['id'][i, j] = ENTITY_IDS[e[0]] if e is not None else 0
            avatar_id = grids[i, :, :, 3].view(-1).max().item()
            targets['id'][i, 3] = avatar_id
        """

        if self.debug_no_predict_other_ids:
            if self.manuals in ['gpt', 'oracle']:
                # only predict id of avatar
                targets['id'][:, :3] = -1

        targets['reward'] = (rewards.long() + 1) * 2
        targets['done'] = dones

        return targets

    def forward(self,
            manuals,
            true_parsed_manuals,
            grids,
            rewards,
            dones,
            actions,
            new_grids,
            new_rewards,
            new_dones,
            new_state_descriptions,
            select_idxs,
            is_eval=False,
        ):

        if manuals is not None:
            manuals = self.reorder_parsed_manuals(manuals, grids)
        true_parsed_manuals = self.reorder_parsed_manuals(true_parsed_manuals, grids)

        """
        if manuals:
            print(manuals[0])
        else:
            print(manuals)
        """

        targets = self.create_targets(
            new_grids,
            new_rewards,
            new_dones,
            new_state_descriptions,
            true_parsed_manuals
        )

        """
        print(new_grids.shape, targets['loc'].shape)
        print(true_parsed_manuals[0])
        #print(grids[0].sum(dim=-1))
        print(new_grids[0].sum(dim=-1))
        #print(new_grids[0, :, :, 0])
        print(targets['loc'][0])
        print(targets['id'][0])
        print(targets['reward'][0])
        print(targets['done'][0])
        """

        logits, self.mem_states = self.step(
            manuals,
            grids,
            rewards,
            dones,
            actions,
            new_state_descriptions,
            self.mem_states,
            is_eval=is_eval
        )

        # select those that are valid continuations
        if manuals is not None:
            manuals = [manuals[i] for i in select_idxs]
        true_parsed_manuals = [true_parsed_manuals[i] for i in select_idxs]
        for k in logits:
            logits[k] = logits[k][select_idxs]
            targets[k] = targets[k][select_idxs]

        if not is_eval:
            for k in logits:
                if 'latent_' in k:
                    continue
                normalizer = math.prod(targets[k].shape)
                logits_flat = logits[k].flatten(0, logits[k].dim() - 2)
                targets_flat = targets[k].flatten()
                self.loss[k] += F.cross_entropy(
                    logits_flat, targets_flat, reduction='sum', ignore_index=-1) / normalizer

            self.backprop_count += 1

        if manuals is not None and self.debug_no_predict_other_ids:
            for i, triplet in enumerate(manuals):
                for j, e in enumerate(triplet):
                    if e is None:
                        id = 0
                    elif e[0] not in ENTITY_IDS:
                        id = 1
                    else:
                        id = ENTITY_IDS[e[0]]
                    for k in range(logits['id'].shape[-1]):
                        if k != id:
                            logits['id'][i, j, k] = float('-inf')

        logits['latent_loc'] = logits['latent'][:, :8]
        logits['latent_id'] = logits['latent'][:, 8:12]
        logits['latent_reward'] = logits['latent'][:, 12]
        logits['latent_done'] = logits['latent'][:, 13]

        targets['latent_loc'] = targets['latent'][:, :8]
        targets['latent_id'] = targets['latent'][:, 8:12]
        targets['latent_reward'] = targets['latent'][:, 12]
        targets['latent_done'] = targets['latent'][:, 13]

        return logits, targets

    """
    def evaluate(self,
            manuals,
            grids,
            rewards,
            dones,
            actions,
            new_state_descriptions,
            select_idxs
        ):

        if self.manuals == 'gpt':
            manuals = self.reorder_parsed_manuals(manuals, grids)

        logits, self.mem_states = self.step(
            manuals,
            grids,
            rewards,
            dones,
            actions,
            new_state_descriptions,
            self.mem_states,
            is_eval=True
        )

        pred_dict = {
            'grid': [],
            'reward': [],
            'done': []
        }
        for i in range(preds.shape[0]):
            description = self.tokenizer.decode(preds[i].tolist()).split()
            new_grid, new_reward, new_done = self._description_to_state(description)
            pred_dict['grid'].append(new_grid)
            pred_dict['reward'].append(new_reward)
            pred_dict['done'].append(new_done)

        pred_dict['grid'] = torch.stack(pred_dict['grid'])
        pred_dict['reward'] = torch.tensor(pred_dict['reward']).to(self.device).float()
        pred_dict['done'] = torch.tensor(pred_dict['done']).to(self.device).long()

        return logit_dict, target_dict, pred_dict

    def _description_to_state(self, d):
        grid = torch.zeros((H, W, GRID_CHANNELS)).to(self.device)
        for c in range(4):
            h = d[c * 2].replace('row_', '')
            w = d[c * 2 + 1].replace('col_', '')
            id = int(d[8 + c].replace('id_', ''))
            if h != 'none' and w != 'none':
                grid[int(h), int(w), c] = id
        reward = (float(d[12].replace('reward_', '')) / 2) - 1
        done = int(d[13] == 'done')
        return grid, reward, done
    """
