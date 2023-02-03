import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from world_model.modules import Encoder, Decoder
from world_model.utils import convert_obs_to_multilabel, convert_multilabel_to_emb, convert_prob_to_multilabel

class WorldModel(nn.Module):
    def __init__(self, key_type, key_dim, val_type, val_dim, latent_size, hidden_size, learning_rate, loss_type, pred_multilabel_threshold, refine_pred_multilabel, device):
        super().__init__()

        emb_dim = 17 + val_dim

        self.latent_size = latent_size 
        self.hidden_size = hidden_size

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

        self.encoder = Encoder(emb_dim, latent_size).to(device)
        self.decoder = Decoder(emb_dim, latent_size).to(device)
        self.lstm = nn.LSTM(latent_size + 5, hidden_size).to(device)
        self.projection = nn.Linear(in_features=hidden_size, out_features=latent_size).to(device)
        self.detector = nn.Sequential(
            nn.Conv2d(in_channels=emb_dim, out_channels=(emb_dim + 17) // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=(emb_dim + 17) // 2, out_channels=17, kernel_size=1, stride=1),
        ).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.real_step_count = 0
        self.imag_step_count = 0
        self.real_loss_total = 0
        self.imag_loss_total = 0

        self.real_tp = torch.zeros(17, dtype=int, device=device)
        self.real_fn = torch.zeros(17, dtype=int, device=device)
        self.real_fp = torch.zeros(17, dtype=int, device=device)
        self.real_tn = torch.zeros(17, dtype=int, device=device)

        self.imag_tp = torch.zeros(17, dtype=int, device=device)
        self.imag_fn = torch.zeros(17, dtype=int, device=device)
        self.imag_fp = torch.zeros(17, dtype=int, device=device)
        self.imag_tn = torch.zeros(17, dtype=int, device=device)

        self.real_dists = []
        self.imag_dists = []

        self.device = device

        self.loss_type = loss_type
        if self.loss_type == "binary":
            self.pos_weight = 10*torch.ones(17, device=device)
            self.pos_weight[0] = 3 / 100
        elif self.loss_type == "cross":
            self.cls_weight = torch.ones(17, device=device)
            self.cls_weight[0] = 3 / 100
            # sc_weight = 3
            # mc_weight = 1
            # avatar_weight = 3 / 8
            # empty_weight = 3 / 96
            # self.cls_weight = torch.tensor([
            #     empty_weight, 0, 
            #     mc_weight, sc_weight, sc_weight, sc_weight, sc_weight, mc_weight, 
            #     mc_weight, mc_weight, sc_weight, mc_weight, mc_weight, sc_weight,
            #     0, avatar_weight, 1,
            # ], device=device)
        else:
            raise NotImplementedError

        self.pred_multilabel_threshold = pred_multilabel_threshold
        self.refine_pred_multilabel = refine_pred_multilabel

        self.vis_logs_reset()

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

    def encode(self, emb):
        return self.encoder(emb.permute(2, 0, 1))

    def decode(self, latent):
        return self.decoder(latent).permute(1, 2, 0)

    def detect(self, emb):
        return self.detector(emb.permute(2, 0, 1)).permute(1, 2, 0)
        
    def forward(self, multilabel, text, ground_truth, action, lstm_states):
        latent = self.encode(convert_multilabel_to_emb(multilabel, text, ground_truth, self))
        action = F.one_hot(torch.tensor(action, device=multilabel.device), num_classes=5)
        lstm_in = torch.cat((latent, action), dim=-1).unsqueeze(0)
        lstm_out, (hidden_state, cell_state) = self.lstm(lstm_in, lstm_states)
        pred_latent = self.projection(lstm_out.squeeze(0))
        pred_logit = self.detect(self.decode(pred_latent))
        return pred_logit, (hidden_state, cell_state)

    def loss(self, logit, prob):
        if self.loss_type == "binary":
            loss = F.binary_cross_entropy_with_logits(logit, prob, pos_weight=self.pos_weight)   
        elif self.loss_type == "cross":
            loss = F.cross_entropy(logit.flatten(0, 1), prob.flatten(0, 1), weight=self.cls_weight)
        else:
            raise NotImplementedError
        return loss

    def logit_to_prob(self, logit):
        if self.loss_type == "binary":
            prob = torch.sigmoid(logit)
        elif self.loss_type == "cross":
            prob = F.softmax(logit, dim=-1)
        else:
            raise NotImplementedError
        return prob

    def multilabel_to_prob(self, multilabel):
        if self.loss_type == "binary":
            prob = multilabel.float()
        elif self.loss_type == "cross":
            prob = multilabel / multilabel.sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        return prob

    def dist(self, pred_multilabel, multilabel):
        nonzeros = torch.nonzero(multilabel[..., 1:])                                           # N x 3
        max_hams = torch.sum(torch.maximum(nonzeros[..., :-1], 9 - nonzeros[..., :-1]), dim=-1) # N

        pred_nonzeros = torch.nonzero(pred_multilabel[..., 1:])                                 # M x 3
        if len(pred_nonzeros > 0):
            comparisons = nonzeros[:, None] - pred_nonzeros[None]                               # N x M x 3
            hams = torch.sum(torch.abs(comparisons[..., :-1]), dim=-1)                          # N x M
            id_diffs = comparisons[..., -1]                                                     # N x M
            hams[id_diffs > 0] = -1
            hams = hams.max(dim=-1).values                                                      # N
            hams[hams < 0] = max_hams[hams < 0]
        else:
            hams = max_hams
        return hams.float().mean()
        
    def real_state_reset(self, init_obs):
        self.real_hidden_state = torch.zeros((1, self.hidden_size), device=self.device)
        self.real_cell_state = torch.zeros((1, self.hidden_size), device=self.device)
        self.real_entity_ids = torch.max(init_obs[..., :-1].flatten(start_dim=0, end_dim=1), dim=0).values

    def imag_state_reset(self, init_obs):
        self.imag_hidden_state = torch.zeros((1, self.hidden_size), device=self.device)
        self.imag_cell_state = torch.zeros((1, self.hidden_size), device=self.device)
        self.imag_old_multilabel = convert_obs_to_multilabel(init_obs)
        self.imag_entity_ids = torch.max(init_obs[..., :-1].flatten(start_dim=0, end_dim=1), dim=0).values

    def real_state_detach(self):
        self.real_hidden_state = self.real_hidden_state.detach()
        self.real_cell_state = self.real_cell_state.detach()

    def imag_state_detach(self):
        self.imag_hidden_state = self.imag_hidden_state.detach()
        self.imag_cell_state = self.imag_cell_state.detach()
        self.imag_old_multilabel = self.imag_old_multilabel.detach()

    def real_step(self, old_obs, text, ground_truth, action, obs):
        self.real_step_count += 1
        old_multilabel = convert_obs_to_multilabel(old_obs)
        multilabel = convert_obs_to_multilabel(obs)
        prob = self.multilabel_to_prob(multilabel)

        pred_logit, (self.real_hidden_state, self.real_cell_state) = self.forward(old_multilabel, text, ground_truth, action, (self.real_hidden_state, self.real_cell_state))
        self.real_loss_total += self.loss(pred_logit, prob)
        with torch.no_grad():
            pred_prob = self.logit_to_prob(pred_logit)
            self.true_real_probs.append(prob.cpu())
            self.pred_real_probs.append(pred_prob.cpu())

            pred_multilabel = convert_prob_to_multilabel(pred_prob, self.pred_multilabel_threshold, self.refine_pred_multilabel, self.real_entity_ids)
            self.true_real_multilabels.append(multilabel.cpu())
            self.pred_real_multilabels.append(pred_multilabel.cpu())

            confusion = pred_multilabel / multilabel # 1 -> tp, 0 -> fn, inf -> fp, nan -> tn
            self.real_tp += torch.sum(confusion == 1, dim=(0, 1))
            self.real_fn += torch.sum(confusion == 0, dim=(0, 1))
            self.real_fp += torch.sum(confusion == float('inf'), dim=(0, 1))
            self.real_tn += torch.sum(torch.isnan(confusion), dim=(0, 1))
            self.real_dists.append(self.dist(pred_multilabel, multilabel))

    def imag_step(self, text, ground_truth, action, obs):
        self.imag_step_count += 1
        old_multilabel = self.imag_old_multilabel
        multilabel = convert_obs_to_multilabel(obs)
        prob = self.multilabel_to_prob(multilabel)

        pred_logit, (self.imag_hidden_state, self.imag_cell_state) = self.forward(old_multilabel, text, ground_truth, action, (self.imag_hidden_state, self.imag_cell_state))
        self.imag_loss_total += self.loss(pred_logit, prob)
        with torch.no_grad():
            pred_prob = self.logit_to_prob(pred_logit)
            self.true_imag_probs.append(prob.cpu())
            self.pred_imag_probs.append(pred_prob.cpu())

            pred_multilabel = convert_prob_to_multilabel(pred_prob, self.pred_multilabel_threshold, self.refine_pred_multilabel, self.imag_entity_ids)
            self.imag_old_multilabel = pred_multilabel
            self.true_imag_multilabels.append(multilabel.cpu())
            self.pred_imag_multilabels.append(pred_multilabel.cpu())

            confusion = pred_multilabel / multilabel # 1 -> tp, 0 -> fn, inf -> fp, nan -> tn
            self.imag_tp += torch.sum(confusion == 1, dim=(0, 1))
            self.imag_fn += torch.sum(confusion == 0, dim=(0, 1))
            self.imag_fp += torch.sum(confusion == float('inf'), dim=(0, 1))
            self.imag_tn += torch.sum(torch.isnan(confusion), dim=(0, 1))
            self.imag_dists.append(self.dist(pred_multilabel, multilabel))

    def real_loss_update(self):
        self.optimizer.zero_grad()
        real_loss_mean = self.real_loss_total / self.real_step_count
        real_loss_mean.backward()
        self.optimizer.step()
        return real_loss_mean.item()

    def imag_loss_update(self):
        self.optimizer.zero_grad()
        imag_loss_mean = self.imag_loss_total / self.imag_step_count
        imag_loss_mean.backward()
        self.optimizer.step()
        return imag_loss_mean.item()

    def real_loss_and_metrics_reset(self):
        recall = self.real_tp / (self.real_tp + self.real_fn)
        precision = self.real_tp / (self.real_tp + self.real_fp)
        f1 = (2 * recall * precision) / (recall + precision)
        metrics = {
            'real_loss': self.real_loss_total.item() / self.real_step_count,
            'real_recall_sprite': recall[1:].nanmean(),
            'real_precision_sprite': precision[1:].nanmean(),
            'real_f1_sprite': f1[1:].nanmean(),
            'real_recall_entity': recall[1:15].nanmean(),
            'real_precision_entity': precision[1:15].nanmean(),
            'real_f1_entity': f1[1:15].nanmean(),
            'real_recall_avatar': recall[15:17].nanmean(),
            'real_precision_avatar': precision[15:17].nanmean(),
            'real_f1_avatar': f1[15:17].nanmean()
        }
        metrics.update({f'real_recall_{i}': recall[i] for i in range(len(recall))})
        metrics.update({f'real_precision_{i}': precision[i] for i in range(len(precision))})
        metrics.update({f'real_f1_{i}': f1[i] for i in range(len(f1))})
        metrics.update({'real_distance': sum(self.real_dists)/len(self.real_dists)})
        
        self.real_step_count = 0
        self.real_loss_total = 0
        self.real_tp = torch.zeros(17, dtype=int, device=self.device)
        self.real_fn = torch.zeros(17, dtype=int, device=self.device)
        self.real_fp = torch.zeros(17, dtype=int, device=self.device)
        self.real_tn = torch.zeros(17, dtype=int, device=self.device)
        self.real_dists = []

        return metrics

    def imag_loss_and_metrics_reset(self):
        recall = self.imag_tp / (self.imag_tp + self.imag_fn)
        precision = self.imag_tp / (self.imag_tp + self.imag_fp)
        f1 = (2 * recall * precision) / (recall + precision)
        metrics = {
            'imag_loss': self.imag_loss_total.item() / self.imag_step_count,
            'imag_recall_sprite': recall[1:].nanmean(),
            'imag_precision_sprite': precision[1:].nanmean(),
            'imag_f1_sprite': f1[1:].nanmean(),
            'imag_recall_entity': recall[1:15].nanmean(),
            'imag_precision_entity': precision[1:15].nanmean(),
            'imag_f1_entity': f1[1:15].nanmean(),
            'imag_recall_avatar': recall[15:17].nanmean(),
            'imag_precision_avatar': precision[15:17].nanmean(),
            'imag_f1_avatar': f1[15:17].nanmean()
        }
        metrics.update({f'imag_recall_{i}': recall[i] for i in range(len(recall))})
        metrics.update({f'imag_precision_{i}': precision[i] for i in range(len(precision))})
        metrics.update({f'imag_f1_{i}': f1[i] for i in range(len(f1))})
        metrics.update({'imag_distance': sum(self.imag_dists)/len(self.imag_dists)})
        
        self.imag_step_count = 0
        self.imag_loss_total = 0
        self.imag_tp = torch.zeros(17, dtype=int, device=self.device)
        self.imag_fn = torch.zeros(17, dtype=int, device=self.device)
        self.imag_fp = torch.zeros(17, dtype=int, device=self.device)
        self.imag_tn = torch.zeros(17, dtype=int, device=self.device)
        self.imag_dists = []

        return metrics

    def vis_logs_reset(self):
        self.true_real_probs = []
        self.pred_real_probs = []
        self.true_real_multilabels = []
        self.pred_real_multilabels = []
        self.true_imag_probs = []
        self.pred_imag_probs = []
        self.true_imag_multilabels = []
        self.pred_imag_multilabels = []
    