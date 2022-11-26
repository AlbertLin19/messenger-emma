import torch
import torch.nn.functional as F

def convert_obs_to_multilabel(obs):
    return torch.sum(F.one_hot(obs, num_classes=17), dim=-2)

def convert_multilabel_to_emb(multilabel, sprite_emb):
    embs = sprite_emb(torch.arange(17, device=multilabel.device))*multilabel[..., None]
    return torch.sum(embs, dim=-2) / torch.sum(multilabel, dim=-1, keepdim=True)
