import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from offline_training.batched_world_model.modules import BatchedEncoder, BatchedDecoder
from offline_training.batched_world_model.utils import batched_convert_grid_to_multilabel, batched_convert_multilabel_to_emb, batched_convert_prob_to_multilabel

class BatchedWorldModel(nn.Module):
    def __init__(self, key_type, key_dim, val_type, val_dim, latent_size, hidden_size, batch_size, learning_rate, prediction_type, pred_multilabel_threshold, refine_pred_multilabel, device):
        super().__init__()

        emb_dim = val_dim # it used to be 17 + val_dim, since torch.cat((multilabel, entity_values), dim=-1) B x 10 x 10 x (17 + val_dim); now, model learns values for empty and avatar, so B x 10 x 10 x val_dim

        self.latent_size = latent_size 
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.key_type = key_type
        self.val_type = val_type

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

        self.empty_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))
        self.avatar_no_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))
        self.avatar_with_message_val_emb = torch.nn.parameter.Parameter(torch.randn(val_dim))
        if val_type == "oracle":
            pass 
        elif val_type == "emma":
            self.txt_val = nn.Linear(768, val_dim).to(device)
            self.scale_val = nn.Sequential(
                nn.Linear(768, 1),
                nn.Softmax(dim=-2)
            ).to(device)
        elif val_type == "emma-mlp_scale":
            self.txt_val = nn.Linear(768, val_dim).to(device)
            self.scale_val = nn.Sequential(
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Linear(384, 1),
                nn.Softmax(dim=-2)
            ).to(device)
        else:
            raise NotImplementedError

        self.encoder = BatchedEncoder(emb_dim, latent_size).to(device)
        self.lstm = nn.LSTM(latent_size + 5, hidden_size).to(device)
        self.projection = nn.Linear(in_features=hidden_size, out_features=latent_size).to(device)
        if prediction_type == "location":
            self.nonexistence = nn.Linear(in_features=latent_size, out_features=17).to(device)
        self.decoder = BatchedDecoder(emb_dim, latent_size).to(device)
        self.detector = nn.Sequential(
            nn.Conv2d(in_channels=emb_dim, out_channels=emb_dim//2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=emb_dim//2, out_channels=17, kernel_size=1, stride=1),
        ).to(device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.real_loss_total = 0
        self.real_backprop_count = 0
        self.imag_loss_total = 0
        self.imag_backprop_count = 0
        
        self.device = device

        self.is_relevant_cls = torch.ones(17, dtype=bool, device=device)
        self.prediction_type = prediction_type
        if self.prediction_type == "existence":
            self.pos_weight = 10*torch.ones(17, device=device)
            self.pos_weight[0] = 3 / 100
            self.is_relevant_cls[torch.tensor([0, 1, 14])] = False
        elif self.prediction_type == "class":
            self.cls_weight = torch.ones(17, device=device)
            self.cls_weight[0] = 3 / 100
            self.is_relevant_cls[torch.tensor([1, 14])] = False
        elif self.prediction_type == "location":
            self.loc_weight = torch.ones(101, device=device)
            self.loc_weight[-1] = 1 / 4
            self.is_relevant_cls[torch.tensor([0, 1, 14])] = False
        else:
            raise NotImplementedError
        self.relevant_cls_idxs = self.is_relevant_cls.argwhere().squeeze(-1)

        self.pred_multilabel_threshold = pred_multilabel_threshold
        self.refine_pred_multilabel = refine_pred_multilabel

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
        
    def forward(self, multilabels, manuals, ground_truths, actions, lstm_states):
        latents = self.encode(batched_convert_multilabel_to_emb(multilabels, manuals, ground_truths, self))
        actions = F.one_hot(actions, num_classes=5)
        lstm_ins = torch.cat((latents, actions), dim=-1).unsqueeze(0)
        lstm_outs, (hidden_states, cell_states) = self.lstm(lstm_ins, lstm_states)
        pred_latents = self.projection(lstm_outs.squeeze(0))
        pred_nonexistence_logits = None
        if self.prediction_type == "location":
            pred_nonexistence_logits = self.nonexistence(pred_latents)
        pred_logits = self.detect(self.decode(pred_latents))
        return (pred_logits, pred_nonexistence_logits), (hidden_states, cell_states)

    def loss(self, logits, nonexistence_logits, probs):
        if self.prediction_type == "existence":
            loss = F.binary_cross_entropy_with_logits(logits, probs, pos_weight=self.pos_weight)   
        elif self.prediction_type == "class":
            loss = F.cross_entropy(logits.flatten(0, 2), probs.flatten(0, 2), weight=self.cls_weight)
        elif self.prediction_type == "location":
            all_logits = torch.cat((logits.permute(0, 3, 1, 2).flatten(2, 3), nonexistence_logits.unsqueeze(-1)), dim=-1).flatten(0, 1)
            nonexistence_probs = 1.0*(torch.sum(probs, dim=(1, 2)) <= 0)
            all_probs = torch.cat((probs.permute(0, 3, 1, 2).flatten(2, 3), nonexistence_probs.unsqueeze(-1)), dim=-1).flatten(0, 1)
            loss = F.cross_entropy(all_logits, all_probs, weight=self.loc_weight)
        else:
            raise NotImplementedError
        return loss

    def logit_to_prob(self, logits, nonexistence_logits):
        nonexistence_probs = None
        if self.prediction_type == "existence":
            probs = torch.sigmoid(logits)
        elif self.prediction_type == "class":
            probs = F.softmax(logits, dim=-1)
        elif self.prediction_type == "location":
            all_logits = torch.cat((logits.permute(0, 3, 1, 2).flatten(2, 3), nonexistence_logits.unsqueeze(-1)), dim=-1)
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

    def real_state_reset(self, init_grids, news):
        if torch.all(news):
            self.real_hidden_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.real_cell_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.real_entity_ids = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values
        else:
            init_grids = init_grids[news]
            self.real_hidden_states[:, news] = torch.zeros((1, 1, self.hidden_size), device=self.device)
            self.real_cell_states[:, news] = torch.zeros((1, 1, self.hidden_size), device=self.device)
            self.real_entity_ids[news] = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values      

    def imag_state_reset(self, init_grids, news):
        if torch.all(news):
            self.imag_hidden_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.imag_cell_states = torch.zeros((1, self.batch_size, self.hidden_size), device=self.device)
            self.imag_old_multilabels = batched_convert_grid_to_multilabel(init_grids)
            self.imag_entity_ids = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values
        else:
            init_grids = init_grids[news]
            self.imag_hidden_states[:, news] = torch.zeros((1, 1, self.hidden_size), device=self.device)
            self.imag_cell_states[:, news] = torch.zeros((1, 1, self.hidden_size), device=self.device)
            self.imag_old_multilabels[news] = batched_convert_grid_to_multilabel(init_grids)
            self.imag_entity_ids[news] = torch.max(init_grids[..., :-1].flatten(start_dim=1, end_dim=2), dim=1).values

    def real_state_detach(self):
        self.real_hidden_states = self.real_hidden_states.detach()
        self.real_cell_states = self.real_cell_states.detach()

    def imag_state_detach(self):
        self.imag_hidden_states = self.imag_hidden_states.detach()
        self.imag_cell_states = self.imag_cell_states.detach()
        self.imag_old_multilabels = self.imag_old_multilabels.detach()

    def real_step(self, old_grids, manuals, ground_truths, actions, grids, do_backprops):
        backprop_idxs = do_backprops.argwhere().squeeze(-1)
        old_multilabels = batched_convert_grid_to_multilabel(old_grids)
        multilabels = batched_convert_grid_to_multilabel(grids)
        probs = self.multilabel_to_prob(multilabels)

        (pred_logits, pred_nonexistence_logits), (self.real_hidden_states, self.real_cell_states) = self.forward(old_multilabels, manuals, ground_truths, actions, (self.real_hidden_states, self.real_cell_states))
        loss = self.loss(pred_logits[backprop_idxs][..., self.relevant_cls_idxs], pred_nonexistence_logits[backprop_idxs][..., self.relevant_cls_idxs], probs[backprop_idxs][..., self.relevant_cls_idxs])
        n_backprops = torch.sum(do_backprops)
        self.real_loss_total += n_backprops*loss
        self.real_backprop_count += n_backprops

    def imag_step(self, manuals, ground_truths, actions, grids, do_backprops):
        backprop_idxs = do_backprops.argwhere().squeeze(-1)
        old_multilabels = self.imag_old_multilabels
        multilabels = batched_convert_grid_to_multilabel(grids)
        probs = self.multilabel_to_prob(multilabels)

        (pred_logits, pred_nonexistence_logits), (self.imag_hidden_states, self.imag_cell_states) = self.forward(old_multilabels, manuals, ground_truths, actions, (self.imag_hidden_states, self.imag_cell_states))
        loss = self.loss(pred_logits[backprop_idxs][..., self.relevant_cls_idxs], pred_nonexistence_logits[backprop_idxs][..., self.relevant_cls_idxs], probs[backprop_idxs][..., self.relevant_cls_idxs])
        n_backprops = torch.sum(do_backprops)
        self.imag_loss_total += n_backprops*loss
        self.imag_backprop_count += n_backprops
        with torch.no_grad():
            pred_probs, pred_nonexistence_probs = self.logit_to_prob(pred_logits, pred_nonexistence_logits)
            pred_multilabels = batched_convert_prob_to_multilabel(pred_probs, pred_nonexistence_probs, self.prediction_type, self.pred_multilabel_threshold, self.refine_pred_multilabel, self.imag_entity_ids)
            self.imag_old_multilabels = pred_multilabels

    def real_loss_update(self):
        self.optimizer.zero_grad()
        real_loss_mean = self.real_loss_total / self.real_backprop_count
        real_loss_mean.backward()
        self.optimizer.step()

    def imag_loss_update(self):
        self.optimizer.zero_grad()
        imag_loss_mean = self.imag_loss_total / self.imag_backprop_count
        imag_loss_mean.backward()
        self.optimizer.step()

    def real_loss_reset(self):
        with torch.no_grad():
            real_loss = self.real_loss_total / self.real_backprop_count
        self.real_loss_total = 0
        self.real_backprop_count = 0
        return real_loss.item()

    def imag_loss_reset(self):
        with torch.no_grad():
            imag_loss = self.imag_loss_total / self.imag_backprop_count
        self.imag_loss_total = 0
        self.imag_backprop_count = 0
        return imag_loss.item()