import random
import math
from pprint import pprint
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from transformers import GPT2Config, GPT2LMHeadModel

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
    def reset(self, is_eval=False):

        self.is_eval = is_eval
        self.memory = None
        self.train(not is_eval)
        self.logit_seq = []
        self.target_seq = []

    """
    def make_description_mask(self, tokenizer):
        vocab_size = len(self.tokenizer.get_vocab())
        mask = torch.zeros((12, vocab_size)).bool()
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
        return mask
    """

    def learn(self):

        batch_size = self.logit_seq[0]['reward'].shape[0]

        loss = defaultdict(lambda: 0)
        for logit, target in zip(self.logit_seq, self.target_seq):
            for k in logit:
                logit_flat = logit[k].flatten(0, logit[k].dim() - 2)
                target_flat = target[k].flatten()
                loss[k] += F.cross_entropy(
                    logit_flat,
                    target_flat,
                    reduction='sum',
                    ignore_index=-1
                ) / batch_size

        total_loss = 0
        for k in loss:
            total_loss += loss[k]
            normalizer = math.prod(self.target_seq[0][k].shape)
            loss[k] = loss[k].item() / normalizer

        total_loss = total_loss / self.gradient_accum_iters
        total_loss.backward()

        #print(total_loss.item() / len(self.logit_seq))
        #print(loss['state'] / 16)

        return loss

    def update_params(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


class WorldModel(WorldModelBase):

    def __init__(self, args):

        super().__init__()

        self.random = random.Random(args.seed + 243)
        self.tokenizer = args.description_tokenizer
        self.batch_size = args.batch_size
        self.device = args.device
        self.manuals = args.manuals
        self.gradient_accum_iters = args.gradient_accum_iters

        self.keep_entity_features_for_parsed_manuals = args.keep_entity_features_for_parsed_manuals
        self.debug_latent_loss_only = args.debug_latent_loss_only
        self.debug_zero_latent = args.debug_zero_latent
        self.debug_no_latent_loss = args.debug_no_latent_loss
        self.debug_no_reward_done_input = args.debug_no_reward_done_input
        self.debug_no_latent = args.debug_no_latent
        self.debug_no_predict_other_ids = args.debug_no_predict_other_ids
        self.debug_no_manual_features = args.debug_no_manual_features

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

        transformer_config = GPT2Config.from_json_file(args.transformer_config_file)
        transformer_config.vocab_size = len(self.tokenizer.get_vocab())

        if args.transformer_n_layer != -1:
            transformer_config.n_layer = args.transformer_n_layer
        if args.transformer_n_embd != -1:
            transformer_config.n_embd = args.transformer_n_embd
        if args.transformer_n_head != -1:
            transformer_config.n_head = args.transformer_n_head

        self.transformer = GPT2LMHeadModel(transformer_config)
        #self.transformer = nn.parallel.DistributedDataParallel(self.transformer)
        #self.description_mask = self.make_description_mask(self.tokenizer).to(self.device)

        vocab_size = transformer_config.vocab_size
        transformer_dim = transformer_config.n_embd

        self.action_embeddings = nn.Embedding(5, transformer_dim)
        self.token_embeddings = nn.Embedding(vocab_size, transformer_dim)

        self.before_mem_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_dim, transformer_dim * 4),
            nn.ReLU()
        )

        self.reward_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, 5),
        )

        self.done_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, 2),
        )

        self.type_to_id = {
            'curr_state' : 0,
            'next_state': 1,
            'action': 2
        }
        self.type_embeddings = nn.Embedding(3, transformer_dim)
        self.global_step_embeddings = nn.Embedding(40, transformer_dim)
        self.local_step_embeddings = nn.Embedding(20, transformer_dim)

        # loss accumulation
        self.loss = {
            'loc' : 0,
            'id' : 0,
            'reward': 0,
            'done': 0
        }
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    def embed_grid_with_parsed_manual(self, grids, parsed_manuals):
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

        if self.debug_no_manual_features:
            return positions_embed + ids_embed

        return roles_embed + movements_embed + positions_embed + ids_embed

    def embed_grid_with_embedded_manual(self, grids, embedded_manuals):
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

    def embed_grid_without_manual(self, grids):

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

    def reorder_parsed_manual(self, parsed_manuals, grids):
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

    def add_curr_state_positional_embeddings(self, curr_token_embed, timestep):

        b, l, _ = curr_token_embed.shape

        curr_type = torch.ones((b, l)).to(self.device).long() * self.type_to_id['curr_state']
        curr_type_embed = self.type_embeddings(curr_type)

        curr_global_step = torch.ones_like(curr_type) * timestep
        curr_global_step_embed = self.global_step_embeddings(curr_global_step)

        assert curr_token_embed.shape == curr_type_embed.shape == curr_global_step_embed.shape

        return curr_token_embed + curr_type_embed + curr_global_step_embed

    def add_next_state_positional_embeddings(self, next_token_embed, timestep):

        b, l, _ = next_token_embed.shape

        next_type = torch.ones((b, l)).to(self.device).long() * self.type_to_id['next_state']
        # first token is action
        next_type[:, 0] = self.type_to_id['action']
        next_type_embed = self.type_embeddings(next_type)

        next_global_step = torch.ones_like(next_type).to(self.device) * (timestep + 1)
        next_global_step_embed = self.global_step_embeddings(next_global_step)

        next_local_step = torch.arange(l).to(self.device).unsqueeze(0).expand(b, -1)
        next_local_step_embed = self.local_step_embeddings(next_local_step)

        assert next_token_embed.shape == next_type_embed.shape == \
                next_global_step_embed.shape == next_local_step_embed.shape

        return next_token_embed + next_type_embed + next_global_step_embed + next_local_step_embed

    def sample_grid(self, action, memory, timestep, batch_size):

        def add_positional_embeddings(embed, global_step, local_step, is_action):
            print(embed.shape)
            type = torch.ones((embed.shape[0], 1)).to(self.device).long()
            if is_action:
                type = type * self.type_to_id['action']
            else:
                type = type * self.type_to_id['next_state']
            type_embed = self.type_embeddings(type)

            global_step = torch.ones_like(type).to(self.device) * (global_step + 1)
            global_step_embed = self.global_step_embeddings(global_step)

            local_step = torch.ones_like(type).to(self.device) * (local_step + 1)
            local_step_embed = self.local_step_embeddings(local_step)

            print(type.tolist()[0], global_step.tolist()[0], local_step.tolist()[0])

            assert embed.shape == type_embed.shape == \
                   global_step_embed.shape == local_step_embed.shape

            return embed + type_embed + global_step_embed + local_step_embed

        token = action.unsqueeze(-1)
        embed = self.token_embeddings(token)
        embed = add_positional_embeddings(embed, timestep, 0, True)

        description_length = 16
        description = []
        for i in range(1, description_length + 1):
            output = self.transformer(
                inputs_embeds=embed,
                past_key_values=memory
            )
            logit = output.logits
            #token = torch.distributions.Categorical(logits=logit).sample()
            print(logit.shape)
            token = logit.max(dim=-1)[1]
            embed = self.token_embeddings(token)
            embed = add_positional_embeddings(embed, timestep, i, False)
            memory = output.past_key_values
            description.append(token)

        grid = torch.zeros((batch_size, H, W, GRID_CHANNELS)).to(self.device)
        description = torch.cat(description, dim=1)
        print(description.shape)
        for i, d in enumerate(description.tolist()):
            tokens = self.tokenizer.decode(d).split()
            print(tokens)
            for j in range(4):
                id = tokens[j * 4 + 1]
                row = tokens[j * 4 + 2]
                col = tokens[j * 4 + 3]
                grid[i, row, col, j] = id

        return grid, memory

    def step(self,
            timestep,
            manual,
            grid,
            action,
            memory,
            sample_grid=False,
            sample_reward_done=False,
            next_state_description=None,
            debug=None
        ):

        if self.manuals in ['gpt', 'oracle']:
            grid_embed = self.embed_grid_with_parsed_manual(grid, manual)
        elif self.manuals == 'embed':
            grid_embed = self.embed_grid_with_embedded_manual(grid, manual)
        elif self.manuals == 'none':
            grid_embed = self.embed_grid_without_manual(grid)
        else:
            raise NotImplementedError

        grid_embed = self.before_mem_projector(self.encoder(grid_embed))

        n_visual_tokens = 4
        curr_token_embed = grid_embed.view(
            grid_embed.shape[0],
            n_visual_tokens,
            grid_embed.shape[1] // n_visual_tokens
        )

        curr_token_embed = self.add_curr_state_positional_embeddings(curr_token_embed, timestep)
        curr_output = self.transformer(
            inputs_embeds=curr_token_embed,
            past_key_values=memory,
            output_hidden_states=True
        )
        curr_memory = curr_output.past_key_values

        if sample_grid:
            next_grid, next_memory = self.sample_grid(action, curr_memory, timestep, grid.shape[0])
            return next_grid, next_memory

        curr_hidden = curr_output.hidden_states[-1]
        curr_hidden = curr_hidden.mean(dim=1)

        logit = {}
        logit['reward'] = self.reward_head(curr_hidden)
        logit['done'] = self.done_head(curr_hidden)

        if sample_reward_done:
            reward = torch.distributions.Categorical(logits=logit['reward']).sample()
            done = torch.distributions.Categorical(logits=logit['done']).sample()
            return reward, done, curr_memory
        else:
            next_token = next_state_description.clone()[:, :-1]
            # set first token to be action
            next_token[:, 0] = action
            next_token_embed = self.token_embeddings(next_token)
            next_state_embed = self.add_next_state_positional_embeddings(next_token_embed, timestep)

            """
            try:
                print('memory', curr_memory[0][0].shape)
            except:
                print('none memory')
                pass
            """

            next_output = self.transformer(
                inputs_embeds=next_token_embed,
                past_key_values=curr_memory
            )
            next_memory = next_output.past_key_values
            logit['state'] = next_output.logits

            if debug is not None:
                tokens = logit['state'][debug].max(-1)[1].tolist()
                print(tokens)
                for j in range(16):
                    print(logit['state'][debug, j, :].topk(5)[1].tolist())
                    print(['%.2f' % x for x in logit['state'][debug, j, :].softmax(dim=0).topk(5)[0]])
                print(self.tokenizer.decode(tokens))
                input()

            return logit, next_memory

    def forward(self,
            timestep,
            manual,
            true_parsed_manual,
            grid,
            reward,
            done,
            action,
            next_state_description,
            mask,
            debug=None
        ):

        if self.manuals in ['gpt', 'oracle']:
            manual = self.reorder_parsed_manual(manual, grid)
        true_parsed_manual = self.reorder_parsed_manual(true_parsed_manual, grid)

        logit, self.memory = self.step(
            timestep,
            manual,
            grid,
            action,
            self.memory,
            next_state_description=next_state_description,
            debug=debug
        )

        target = {}

        target['state'] = next_state_description[:, 1:]
        target['state'] = target['state'].masked_fill(mask.unsqueeze(1), -1)

        target['reward'] = (reward.long() + 1) * 2
        target['reward'] = target['reward'].masked_fill(mask, -1)

        target['done'] = done
        target['done'] = target['done'].masked_fill(mask, -1)

        self.logit_seq.append(logit)
        self.target_seq.append(target)

        return logit, target


class WorldModelEnv:

    def __init__(self, world_model):
        self.world_model = world_model
        self.device = world_model.device

    def reset(self, grid, manual):
        batch_size = grid.shape[0]
        self.t = 0
        self.manual = manual
        self.grid = grid
        self.world_model.reset(is_eval=True)
        self.memory = None

        return grid

    def step(self, action):

        action = torch.tensor(action).to(self.device).long()
        self.grid, self.memory = self.world_model.step(
            self.t,
            self.manual,
            self.grid,
            action,
            self.memory,
            sample_grid=True
        )
        self.t += 1
        reward, done, self.memory = self.world_model.step(
            self.t,
            self.manual,
            self.grid,
            None,
            self.memory,
            sample_reward_done=True
        )

        return self.grid, reward, done

