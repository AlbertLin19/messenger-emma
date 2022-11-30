import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from world_model.modules import Encoder, Decoder
from world_model.utils import convert_obs_to_multilabel, convert_multilabel_to_emb

class WorldModel(nn.Module):
    def __init__(self, emma, val_emb_dim, latent_size, hidden_size, learning_rate, loss_type, device):
        super().__init__()

        emb_dim = emma.emb_dim + val_emb_dim

        self.latent_size = latent_size 
        self.hidden_size = hidden_size

        self.emma = emma
        self.txt_val = nn.Linear(768, val_emb_dim).to(device)
        self.scale_val = nn.Sequential(
            nn.Linear(768, 1),
            nn.Softmax(dim=-2)
        ).to(device)

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
        self.real_loss = 0
        self.imag_loss = 0

        self.real_tp = torch.zeros(17, dtype=int, device=device)
        self.real_fn = torch.zeros(17, dtype=int, device=device)
        self.real_fp = torch.zeros(17, dtype=int, device=device)
        self.real_tn = torch.zeros(17, dtype=int, device=device)

        self.imag_tp = torch.zeros(17, dtype=int, device=device)
        self.imag_fn = torch.zeros(17, dtype=int, device=device)
        self.imag_fp = torch.zeros(17, dtype=int, device=device)
        self.imag_tn = torch.zeros(17, dtype=int, device=device)

        self.device = device

        self.loss_type = loss_type
        if self.loss_type == "binary_cross_entropy":
            self.pos_weight = 10*torch.ones(17, device=device)
            self.pos_weight[0] = 3 / 100
        elif self.loss_type == "cross_entropy":
            self.pos_threshold = 0.4
            self.cls_weight = torch.ones(17, device=device)
            self.cls_weight[0] = 3 / 100
        else:
            raise NotImplementedError

        # EPISODE LOGGING
        self.true_real_grids = []
        self.true_imag_grids = []
        self.pred_real_grids = []
        self.pred_imag_grids = []

    def encode(self, emb):
        return self.encoder(emb.permute(2, 0, 1))

    def decode(self, latent):
        return self.decoder(latent).permute(1, 2, 0)

    def detect(self, emb):
        return self.detector(emb.permute(2, 0, 1)).permute(1, 2, 0)
        
    def real_state_reset(self):
        self.real_hidden_state = torch.zeros((1, self.hidden_size), device=self.device)
        self.real_cell_state = torch.zeros((1, self.hidden_size), device=self.device)

    def imag_state_reset(self, init_obs):
        self.imag_hidden_state = torch.zeros((1, self.hidden_size), device=self.device)
        self.imag_cell_state = torch.zeros((1, self.hidden_size), device=self.device)
        self.imag_old_multilabel = convert_obs_to_multilabel(init_obs)

    def real_state_detach(self):
        self.real_hidden_state = self.real_hidden_state.detach()
        self.real_cell_state = self.real_cell_state.detach()

    def imag_state_detach(self):
        self.imag_hidden_state = self.imag_hidden_state.detach()
        self.imag_cell_state = self.imag_cell_state.detach()
        self.imag_old_multilabel = self.imag_old_multilabel.detach()

    def real_step(self, old_obs, text, action, obs):
        old_multilabel = convert_obs_to_multilabel(old_obs)
        multilabel = convert_obs_to_multilabel(obs)

        old_latent = self.encode(convert_multilabel_to_emb(old_multilabel, text, self))
        action = F.one_hot(torch.tensor(action, device=obs.device), num_classes=5)
        lstm_in = torch.cat((old_latent, action), dim=-1).unsqueeze(0)
        lstm_out, (self.real_hidden_state, self.real_cell_state) = self.lstm(lstm_in, (self.real_hidden_state, self.real_cell_state))
        latent = self.projection(lstm_out.squeeze(0))
        pred_multilabel_logit = self.detect(self.decode(latent))

        if self.loss_type == "binary_cross_entropy":
            self.real_loss += F.binary_cross_entropy_with_logits(pred_multilabel_logit, multilabel.float(), pos_weight=self.pos_weight)
            confusion = (pred_multilabel_logit > 0) / multilabel # 1 -> tp, 0 -> fn, inf -> fp, nan -> tn
        elif self.loss_type == "cross_entropy":
            self.real_loss += F.cross_entropy(pred_multilabel_logit.flatten(0, 1), (multilabel / multilabel.sum(dim=-1, keepdim=True)).flatten(0, 1), weight=self.cls_weight)
            confusion = (F.softmax(pred_multilabel_logit, dim=-1) >= self.pos_threshold) / multilabel # 1 -> tp, 0 -> fn, inf -> fp, nan -> tn
        else:
            raise NotImplementedError
        self.real_tp += torch.sum(confusion == 1, dim=(0, 1))
        self.real_fn += torch.sum(confusion == 0, dim=(0, 1))
        self.real_fp += torch.sum(confusion == float('inf'), dim=(0, 1))
        self.real_tn += torch.sum(torch.isnan(confusion), dim=(0, 1))

        # EPISODE LOGGING
        if self.loss_type == "binary_cross_entropy":
            true_grid = multilabel.cpu() 
            pred_grid = torch.sigmoid(pred_multilabel_logit).detach().cpu()
        elif self.loss_type == "cross_entropy":
            true_grid = (multilabel / multilabel.sum(dim=-1, keepdim=True)).cpu()
            pred_grid = torch.softmax(pred_multilabel_logit, dim=-1).detach().cpu()
        else:
            raise NotImplementedError
        self.true_real_grids.append(true_grid)
        self.pred_real_grids.append(pred_grid)

    def imag_step(self, text, action, obs):
        multilabel = convert_obs_to_multilabel(obs)

        old_latent = self.encode(convert_multilabel_to_emb(self.imag_old_multilabel, text, self))
        action = F.one_hot(torch.tensor(action, device=obs.device), num_classes=5)
        lstm_in = torch.cat((old_latent, action), dim=-1).unsqueeze(0)
        lstm_out, (self.imag_hidden_state, self.imag_cell_state) = self.lstm(lstm_in, (self.imag_hidden_state, self.imag_cell_state))
        latent = self.projection(lstm_out.squeeze(0))
        pred_multilabel_logit = self.detect(self.decode(latent))

        if self.loss_type == "binary_cross_entropy":
            self.imag_old_multilabel = 1*(pred_multilabel_logit > 0)
        elif self.loss_type == "cross_entropy":
            self.imag_old_multilabel = 1*(F.softmax(pred_multilabel_logit, dim=-1) >= self.pos_threshold)

        if self.loss_type == "binary_cross_entropy":
            self.imag_loss += F.binary_cross_entropy_with_logits(pred_multilabel_logit, multilabel.float(), pos_weight=self.pos_weight)
            confusion = (pred_multilabel_logit > 0) / multilabel # 1 -> tp, 0 -> fn, inf -> fp, nan -> tn
        elif self.loss_type == "cross_entropy":
            self.imag_loss += F.cross_entropy(pred_multilabel_logit.flatten(0, 1), (multilabel / multilabel.sum(dim=-1, keepdim=True)).flatten(0, 1), weight=self.cls_weight)
            confusion = (F.softmax(pred_multilabel_logit, dim=-1) >= self.pos_threshold) / multilabel # 1 -> tp, 0 -> fn, inf -> fp, nan -> tn
        else:
            raise NotImplementedError
        self.imag_tp += torch.sum(confusion == 1, dim=(0, 1))
        self.imag_fn += torch.sum(confusion == 0, dim=(0, 1))
        self.imag_fp += torch.sum(confusion == float('inf'), dim=(0, 1))
        self.imag_tn += torch.sum(torch.isnan(confusion), dim=(0, 1))

        # EPISODE LOGGING
        if self.loss_type == "binary_cross_entropy":
            true_grid = multilabel.cpu()
            pred_grid = torch.sigmoid(pred_multilabel_logit).detach().cpu()
        elif self.loss_type == "cross_entropy":
            true_grid = (multilabel / multilabel.sum(dim=-1, keepdim=True)).cpu()
            pred_grid = torch.softmax(pred_multilabel_logit, dim=-1).detach().cpu()
        else:
            raise NotImplementedError
        self.true_imag_grids.append(true_grid)
        self.pred_imag_grids.append(pred_grid)

    def real_loss_update(self):
        self.optimizer.zero_grad()
        self.real_loss.backward()
        self.optimizer.step()
        return self.real_loss.item()

    def imag_loss_update(self):
        self.optimizer.zero_grad()
        self.imag_loss.backward()
        self.optimizer.step()
        return self.imag_loss.item()

    def real_loss_and_metrics_reset(self):
        recall = self.real_tp / (self.real_tp + self.real_fn)
        precision = self.real_tp / (self.real_tp + self.real_fp)
        f1 = (2 * recall * precision) / (recall + precision)
        metrics = {
            'real_loss': self.real_loss.item(),
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
        
        self.real_loss = 0
        self.real_tp = torch.zeros(17, dtype=int, device=self.device)
        self.real_fn = torch.zeros(17, dtype=int, device=self.device)
        self.real_fp = torch.zeros(17, dtype=int, device=self.device)
        self.real_tn = torch.zeros(17, dtype=int, device=self.device)

        return metrics

    def imag_loss_and_metrics_reset(self):
        recall = self.imag_tp / (self.imag_tp + self.imag_fn)
        precision = self.imag_tp / (self.imag_tp + self.imag_fp)
        f1 = (2 * recall * precision) / (recall + precision)
        metrics = {
            'imag_loss': self.imag_loss.item(),
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
        
        self.imag_loss = 0
        self.imag_tp = torch.zeros(17, dtype=int, device=self.device)
        self.imag_fn = torch.zeros(17, dtype=int, device=self.device)
        self.imag_fp = torch.zeros(17, dtype=int, device=self.device)
        self.imag_tn = torch.zeros(17, dtype=int, device=self.device)

        return metrics

    def episode_logs_reset(self):
        self.true_real_grids = []
        self.true_imag_grids = []
        self.pred_real_grids = []
        self.pred_imag_grids = []
    