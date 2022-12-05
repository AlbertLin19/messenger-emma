import torch
import torch.nn.functional as F

def convert_obs_to_multilabel(obs):
    multilabel = torch.sum(F.one_hot(obs, num_classes=17), dim=-2)
    multilabel[..., 0] = torch.sum(obs, dim=-1) < 1
    return multilabel

def ground(text, emma):
    query = emma.sprite_emb(torch.arange(17, device=emma.device)) # 17 x sprite_emb_dim

    # Attention-based text representation        
    key = emma.txt_key(text)
    key_scale = emma.scale_key(text) # (num sent, sent_len, 1)
    key = key * key_scale
    key = torch.sum(key, dim=1) # num sent x key_emb_dim
    
    kq = query @ key.t() # dot product attention (17 x num sent)
    mask = (kq != 0) # keep zeroed-out entries zero
    kq = kq / emma.attn_scale # scale to prevent vanishing grads
    weights = F.softmax(kq, dim=-1) * mask # (17 x num sent)
    
    return weights

def convert_multilabel_to_emb(multilabel, text, world_model):
    value = world_model.txt_val(text)
    val_scale = world_model.scale_val(text)
    value = value * val_scale
    value = torch.sum(value, dim=1) # num sent x val_emb_dim

    weights = ground(text, world_model)
    entity_values = torch.mean(weights.unsqueeze(-1) * value, dim=-2) # (17 x val_emb_dim)
    entity_values[15:17] = torch.tensor([0], device=world_model.device)
    entity_values = entity_values*multilabel[..., None] # (10 x 10 x 17 x val_emb_dim)
    entity_value = torch.sum(entity_values, dim=-2)

    return torch.cat((multilabel, entity_value), dim=-1)

def key_attend(text, emma):
    # Attention-based text representation        
    key_scale = emma.scale_key(text) # (num sent, sent_len, 1)
    return key_scale

def value_attend(text, emma):
    val_scale = emma.scale_val(text)
    return val_scale

def convert_prob_to_multilabel(prob, threshold, refine, entity_ids):
    multilabel = 1*(prob > torch.maximum(prob[..., 0:1], torch.tensor([threshold], device=prob.device)))
    if refine:
        multilabel = multilabel*(prob >= torch.amax(prob, dim=(0, 1)))
        multilabel[..., 15:17] = (prob[..., 15:17] >= torch.max(prob[..., 15:17]))
        multilabel[..., :15] = multilabel[..., :15]*(F.one_hot(entity_ids, num_classes=15).sum(dim=0))
    multilabel[..., 0] = (multilabel.sum(dim=-1) < 1)
    return multilabel