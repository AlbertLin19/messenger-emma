import torch
import torch.nn.functional as F

from messenger.envs.config import NPCS

ENTITY_NAMES = {entity.id: entity.name for entity in NPCS}
def get_NPCS():
    return ENTITY_NAMES.copy()

ENTITY_IDS = {entity.name: entity.id for entity in NPCS}
MOVEMENT_TYPES = {
    "chaser": 0,
    "fleeing": 1,
    "immovable": 2,
}

def convert_obs_to_multilabel(obs):
    multilabel = torch.sum(F.one_hot(obs, num_classes=17), dim=-2)
    multilabel[..., 0] = torch.sum(obs, dim=-1) < 1
    return multilabel

def ground(text, ground_truth, world_model):
    query = world_model.sprite_emb(torch.arange(17, device=world_model.device)) # 17 x sprite_emb_dim
    if world_model.key_type == "oracle":
        key = F.one_hot(torch.tensor([ENTITY_IDS[truth[0]] for truth in ground_truth], device=world_model.device), num_classes=17).float()
    elif world_model.key_type == "emma" or world_model.key_type == "emma-mlp_scale":
        # Attention-based text representation        
        key = world_model.txt_key(text)
        key_scale = world_model.scale_key(text) # (num sent, sent_len, 1)
        key = key * key_scale
        key = torch.sum(key, dim=1) # num sent x key_emb_dim
    else:
        raise NotImplementedError
        
    kq = query @ key.t() # dot product attention (17 x num sent)
    if world_model.key_type == "oracle":
        return kq
    elif world_model.key_type == "emma" or world_model.key_type == "emma-mlp_scale":
        mask = (kq != 0) # keep zeroed-out entries zero
        kq = kq / world_model.attn_scale # scale to prevent vanishing grads
        weights = F.softmax(kq, dim=-1) * mask # (17 x num sent)
    else:
        # for other key_types, should I use world_model.attn_scale?
        raise NotImplementedError
    
    return weights

def convert_multilabel_to_emb(multilabel, text, ground_truth, world_model):
    if world_model.val_type == "oracle":
        value = F.one_hot(torch.tensor([MOVEMENT_TYPES[truth[1]] for truth in ground_truth], device=world_model.device), num_classes=3)
    elif world_model.val_type == "emma":
        value = world_model.txt_val(text)
        val_scale = world_model.scale_val(text)
        value = value * val_scale
        value = torch.sum(value, dim=1) # num sent x val_emb_dim
    else:
        raise NotImplementedError

    weights = ground(text, ground_truth, world_model)
    entity_values = torch.mean(weights.unsqueeze(-1) * value, dim=-2) # (17 x val_emb_dim)
    entity_values[0:1] = torch.tensor([0], device=world_model.device)
    entity_values[15:17] = torch.tensor([0], device=world_model.device)
    entity_values = entity_values*multilabel[..., None] # (10 x 10 x 17 x val_emb_dim)
    entity_value = torch.sum(entity_values, dim=-2)
    return torch.cat((multilabel, entity_value), dim=-1)

def convert_prob_to_multilabel(prob, threshold, refine, entity_ids):
    multilabel = 1*(prob > torch.maximum(prob[..., 0:1], torch.tensor([threshold], device=prob.device)))
    if refine:
        multilabel = multilabel*(prob >= torch.amax(prob, dim=(0, 1)))
        multilabel[..., 15:17] = (prob[..., 15:17] >= torch.max(prob[..., 15:17]))
        multilabel[..., :15] = multilabel[..., :15]*(F.one_hot(entity_ids, num_classes=15).sum(dim=0))
    multilabel[..., 0] = (multilabel.sum(dim=-1) < 1)
    return multilabel