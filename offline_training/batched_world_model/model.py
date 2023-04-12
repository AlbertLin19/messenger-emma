import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from offline_training.batched_world_model.modules import BatchedEncoder, BatchedDecoder
from offline_training.batched_world_model.utils import batched_convert_grid_to_multilabel, batched_convert_multilabel_to_emb, batched_convert_prob_to_multilabel

class BatchedWorldModel(nn.Module):
    def __init__(self, key_type, key_dim, val_type, val_dim, memory_type, latent_size, hidden_size, batch_size, learning_rate, reward_loss_weight, done_loss_weight, prediction_type, pred_multilabel_threshold, refine_pred_multilabel, dropout_prob, dropout_loc, shuffle_ids, device):
        super().__init__()

        self.latent_size = latent_size 
        self.memory_type = memory_type
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.key_type = key_type
        self.val_type = val_type
        self.val_dim = val_dim

        if key_type == "oracle":
            self.sprite_emb = lambda x: F.one_hot(x, num_classes=17).float()

        elif key_type == "emma":
            self.sprite_emb = nn.Embedding(17, key_dim, padding_idx=0).to(device) # sprite embedding layer
            self.attn_scale = np.sqrt(key_dim)
            self.txt_key = nn.Linear(768, key_dim).to(device)
            self.scale_key = nn.Sequential(
                nn.Linear(768, 1),
                nn.Softmax(dim=-2)
            ).to(device)

        elif key_type == "emma-mlp_scale":
            self.sprite_emb = nn.Embedding(17, key_dim, padding_idx=0).to(device) # sprite embedding layer
            self.attn_scale = np.sqrt(key_dim)
            self.txt_key = nn.Linear(768, key_dim).to(device)
            self.scale_key = nn.Sequential(
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Linear(384, 1),
                nn.Softmax(dim=-2)
            ).to(device)

        else:
            raise NotImplementedError

        if val_type == "oracle":
            self.avatar_no_message_val_emb = torch.tensor([0, 0, 0, 0, 0, 0, 1], device=device)
            self.avatar_with_message_val_emb = torch.tensor([0, 0, 0, 0, 0, 0, 1], device=device)
        
        elif val_type == "emma":
            self.avatar_no_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))
            self.avatar_with_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))
        
            self.txt_val = nn.Linear(768, val_dim).to(device)
            self.scale_val = nn.Sequential(
                nn.Linear(768, 1),
                nn.Softmax(dim=-2)
            ).to(device)

        elif val_type == "emma-mlp_scale":
            self.avatar_no_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))
            self.avatar_with_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))
        
            self.txt_val = nn.Linear(768, val_dim).to(device)
            self.scale_val = nn.Sequential(
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
        if self.prediction_type == "existence":
            self.pos_weight = 10*torch.ones(17, device=device)
            self.pos_weight[0] = 3 / 100
            self.relevant_cls_idxs = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], device=device)

        elif self.prediction_type == "class":
            self.cls_weight = torch.ones(17, device=device)
            self.cls_weight[0] = 3 / 100
            self.relevant_cls_idxs = torch.tensor([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], device=device)

        elif self.prediction_type == "location":
            self.loc_weight = torch.ones(101, device=device)
            self.relevant_cls_idxs = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], device=device)

        else:
            raise NotImplementedError

        emb_dim = val_dim + len(self.relevant_cls_idxs)
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout_prob if "input" in dropout_loc else 0),
            BatchedEncoder(emb_dim, latent_size),
            nn.Dropout(p=dropout_prob if "network" in dropout_loc else 0),
        ).to(device)
        if self.memory_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(latent_size + 5, hidden_size),
                nn.ReLU(),
            ).to(device)
        elif self.memory_type == "lstm":
            self.lstm = nn.LSTM(latent_size + 5, hidden_size).to(device)
        else:
            raise NotImplementedError
        self.projection = nn.Linear(in_features=hidden_size, out_features=latent_size).to(device)
        if prediction_type == "location":
            self.nonexistence = nn.Linear(in_features=latent_size, out_features=17).to(device)
        self.decoder = BatchedDecoder(emb_dim, latent_size).to(device)
        self.detector = nn.Sequential(
            nn.Conv2d(in_channels=emb_dim, out_channels=(emb_dim + 17) // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=(emb_dim + 17) // 2, out_channels=17, kernel_size=1, stride=1),
        ).to(device)
        self.reward_head = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=1),
            nn.Flatten(start_dim=0, end_dim=-1),
        ).to(device)
        self.done_head = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=1),
            nn.Flatten(start_dim=0, end_dim=-1),
        ).to(device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.reward_loss_weight = reward_loss_weight
        self.done_loss_weight = done_loss_weight

        self.real_grid_loss_total = 0
        self.real_reward_loss_total = 0
        self.real_done_loss_total = 0
        self.real_backprop_count = 0
        self.imag_grid_loss_total = 0
        self.imag_reward_loss_total = 0
        self.imag_done_loss_total = 0
        self.imag_backprop_count = 0

        self.pred_multilabel_threshold = pred_multilabel_threshold
        self.refine_pred_multilabel = refine_pred_multilabel
        self.shuffle_ids = shuffle_ids        
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
        
    def forward(self, multilabels, manuals, ground_truths, actions, lstm_states, shuffled_ids):
        embeddings = batched_convert_multilabel_to_emb(multilabels, manuals, ground_truths, self)
        if self.shuffle_ids:
            embeddings[..., :len(self.relevant_cls_idxs)] = torch.gather(input=embeddings[..., :len(self.relevant_cls_idxs)], dim=-1, index=shuffled_ids.unsqueeze(1).unsqueeze(1).expand(-1, 10, 10, -1))
        latents = self.encode(embeddings)
        actions = F.one_hot(actions, num_classes=5)
        mem_ins = torch.cat((latents, actions), dim=-1).unsqueeze(0)
        if self.memory_type == "mlp":
            mem_outs = self.mlp(mem_ins)
            hidden_states, cell_states = None, None
        elif self.memory_type == "lstm":
            mem_outs, (hidden_states, cell_states) = self.lstm(mem_ins, lstm_states)
        else:
            raise NotImplementedError
        pred_latents = self.projection(mem_outs.squeeze(0))
        pred_nonexistence_logits = None
        if self.prediction_type == "location":
            pred_nonexistence_logits = self.nonexistence(pred_latents)
        pred_grid_logits = self.detect(self.decode(pred_latents))
        if self.shuffle_ids:
            pred_grid_logits[..., self.relevant_cls_idxs] = torch.gather(input=pred_grid_logits[..., self.relevant_cls_idxs], dim=-1, index=torch.argsort(shuffled_ids, dim=-1).unsqueeze(1).unsqueeze(1).expand(-1, 10, 10, -1))
            pred_nonexistence_logits[..., self.relevant_cls_idxs] = torch.gather(input=pred_nonexistence_logits[..., self.relevant_cls_idxs], dim=-1, index=torch.argsort(shuffled_ids, dim=-1))
        pred_rewards = self.reward_head(mem_outs.squeeze(0))
        pred_done_logits = self.done_head(mem_outs.squeeze(0))
        return ((pred_grid_logits, pred_nonexistence_logits), pred_rewards, pred_done_logits), (hidden_states, cell_states)

    def grid_loss(self, grid_logits, nonexistence_logits, probs):
        if self.prediction_type == "existence":
            loss = F.binary_cross_entropy_with_logits(grid_logits, probs, pos_weight=self.pos_weight)   
        elif self.prediction_type == "class":
            loss = F.cross_entropy(grid_logits.flatten(0, 2), probs.flatten(0, 2), weight=self.cls_weight)
        elif self.prediction_type == "location":
            all_logits = torch.cat((grid_logits.permute(0, 3, 1, 2).flatten(2, 3), nonexistence_logits.unsqueeze(-1)), dim=-1).flatten(0, 1)
            nonexistence_probs = 1.0*(torch.sum(probs, dim=(1, 2)) <= 0)
            all_probs = torch.cat((probs.permute(0, 3, 1, 2).flatten(2, 3), nonexistence_probs.unsqueeze(-1)), dim=-1).flatten(0, 1)
            loss = F.cross_entropy(all_logits, all_probs, weight=self.loc_weight)
        else:
            raise NotImplementedError
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

    def real_state_reset(self, init_grids, idxs=None):
        if idxs is None:
            self.real_hidden_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.real_cell_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.real_entity_ids = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values
            self.real_shuffled_ids = None
            if self.shuffle_ids:
                self.real_shuffled_ids = torch.from_numpy(np.random.default_rng().permuted(np.broadcast_to(np.arange(len(self.relevant_cls_idxs)), (self.batch_size, len(self.relevant_cls_idxs))), axis=-1)).long().to(self.device)
        else:
            init_grids = init_grids[idxs]
            self.real_hidden_states[:, idxs] = 0
            self.real_cell_states[:, idxs] = 0
            self.real_entity_ids[idxs] = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values      
            if self.shuffle_ids:
                self.real_shuffled_ids[idxs] = torch.from_numpy(np.random.default_rng().permuted(np.broadcast_to(np.arange(len(self.relevant_cls_idxs)), (len(idxs), len(self.relevant_cls_idxs))), axis=-1)).long().to(self.device)

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

    def real_state_detach(self):
        self.real_hidden_states = self.real_hidden_states.detach()
        self.real_cell_states = self.real_cell_states.detach()

    def imag_state_detach(self):
        self.imag_hidden_states = self.imag_hidden_states.detach()
        self.imag_cell_states = self.imag_cell_states.detach()
        self.imag_old_multilabels = self.imag_old_multilabels.detach()

    def real_step(self, old_grids, manuals, ground_truths, actions, grids, rewards, dones, backprop_idxs):
        old_multilabels = batched_convert_grid_to_multilabel(old_grids)
        multilabels = batched_convert_grid_to_multilabel(grids)
        probs = self.multilabel_to_prob(multilabels)
        done_probs = dones.float()

        (pred_loc_logits, pred_rewards, pred_done_logits), (self.real_hidden_states, self.real_cell_states) = self.forward(old_multilabels, manuals, ground_truths, actions, (self.real_hidden_states, self.real_cell_states), self.real_shuffled_ids)
        pred_grid_logits, pred_nonexistence_logits = pred_loc_logits
        n_backprops = len(backprop_idxs)
        self.real_grid_loss_total += n_backprops*self.grid_loss(pred_grid_logits[backprop_idxs][..., self.relevant_cls_idxs], pred_nonexistence_logits[backprop_idxs][..., self.relevant_cls_idxs], probs[backprop_idxs][..., self.relevant_cls_idxs])
        self.real_reward_loss_total += n_backprops*self.reward_loss(pred_rewards[backprop_idxs], rewards[backprop_idxs])
        self.real_done_loss_total += n_backprops*self.done_loss(pred_done_logits[backprop_idxs], done_probs[backprop_idxs])
        self.real_backprop_count += n_backprops

        with torch.no_grad():
            pred_grid_probs, pred_nonexistence_probs = self.logit_to_prob(pred_grid_logits, pred_nonexistence_logits)
            pred_multilabels = batched_convert_prob_to_multilabel(pred_grid_probs, pred_nonexistence_probs, self.prediction_type, self.pred_multilabel_threshold, self.refine_pred_multilabel, self.real_entity_ids)
            pred_done_probs = torch.sigmoid(pred_done_logits)
        return (((pred_grid_probs, pred_nonexistence_probs), pred_multilabels), pred_rewards, pred_done_probs), ((probs, multilabels), rewards, done_probs)

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

    def real_loss_update(self):
        self.optimizer.zero_grad()
        real_loss_mean = (self.real_grid_loss_total + self.reward_loss_weight*self.real_reward_loss_total + self.done_loss_weight*self.real_done_loss_total) / self.real_backprop_count
        real_loss_mean.backward()
        self.optimizer.step()

    def imag_loss_update(self):
        self.optimizer.zero_grad()
        imag_loss_mean = (self.imag_grid_loss_total + self.reward_loss_weight*self.imag_reward_loss_total + self.done_loss_weight*self.imag_done_loss_total) / self.imag_backprop_count
        imag_loss_mean.backward()
        self.optimizer.step()

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