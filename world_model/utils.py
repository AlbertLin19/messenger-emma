import torch
import torch.nn.functional as F

def convert_obs_to_multilabel(obs):
    multilabel = torch.sum(F.one_hot(obs, num_classes=17), dim=-2)
    multilabel[..., 0] = torch.sum(obs, dim=-1) < 1
    return multilabel

def convert_multilabel_to_emb(multilabel, text, world_model):
    query = world_model.emma.sprite_emb(torch.arange(17, device=world_model.device)) # 17 x sprite_emb_dim

    # Attention-based text representation        
    key = world_model.emma.txt_key(text)
    key_scale = world_model.emma.scale_key(text) # (num sent, sent_len, 1)
    key = key * key_scale
    key = torch.sum(key, dim=1) # num sent x key_emb_dim
    
    # use world_model custom value
    value = world_model.txt_val(text)
    val_scale = world_model.scale_val(text)
    value = value * val_scale
    value = torch.sum(value, dim=1) # num sent x val_emb_dim

    kq = query @ key.t() # dot product attention (17 x num sent)
    mask = (kq != 0) # keep zeroed-out entries zero
    kq = kq / world_model.emma.attn_scale # scale to prevent vanishing grads
    weights = F.softmax(kq, dim=-1) * mask # (17 x num sent)
    values = torch.mean(weights.unsqueeze(-1) * value, dim=-2) # (17 x val_emb_dim)

    keys = query*multilabel[..., None] # (10 x 10 x 17 x key_emb_dim)
    key = torch.sum(keys, dim=-2) / torch.sum(multilabel, dim=-1, keepdim=True)

    values[15:17] = query[15:17]
    values = values*multilabel[..., None] # (10 x 10 x 17 x val_emb_dim)
    value = torch.sum(values, dim=-2) / torch.sum(multilabel, dim=-1, keepdim=True)

    return torch.cat((key, value), dim=-1)

def ground(text, emma):
    query = emma.sprite_emb(torch.arange(17, device=text.device)) # 17 x sprite_emb_dim

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

def key_attend(text, emma):
    # Attention-based text representation        
    key_scale = emma.scale_key(text) # (num sent, sent_len, 1)
    return key_scale

def value_attend(text, emma):
    val_scale = emma.scale_val(text)
    return val_scale

def convert_prob_to_multilabel(prob, entity_ids):
    multilabel = 1*(prob > prob[..., 0:1])
    multilabel[..., 0] = (multilabel.sum(dim=-1) < 1)
    return multilabel