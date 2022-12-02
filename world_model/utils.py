import torch
import torch.nn.functional as F

def convert_obs_to_multilabel(obs):
    multilabel = torch.sum(F.one_hot(obs, num_classes=17), dim=-2)
    multilabel[..., 0] = torch.logical_not(torch.sum(obs > 0, dim=-1))
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

def attend(text, world_model):
    query = world_model.emma.sprite_emb(torch.arange(17, device=world_model.device)) # 17 x sprite_emb_dim

    # Attention-based text representation        
    key = world_model.emma.txt_key(text)
    key_scale = world_model.emma.scale_key(text) # (num sent, sent_len, 1)
    key = key * key_scale
    key = torch.sum(key, dim=1) # num sent x key_emb_dim
    
    kq = query @ key.t() # dot product attention (17 x num sent)
    mask = (kq != 0) # keep zeroed-out entries zero
    kq = kq / world_model.emma.attn_scale # scale to prevent vanishing grads
    weights = F.softmax(kq, dim=-1) * mask # (17 x num sent)
    
    return weights

def convert_prob_to_multilabel(prob, entity_ids):
    multilabel = torch.zeros((10, 10, 17), device=prob.device)
    multilabel[..., 0] = 1
    for entity_id in entity_ids:
        max_prob = torch.max(prob[..., entity_id])
        argmax_prob = torch.nonzero(prob[..., entity_id] == max_prob)[0]
        if max_prob > prob[argmax_prob[0], argmax_prob[1], 0]:
            multilabel[argmax_prob[0], argmax_prob[1], 0] = 0
            multilabel[argmax_prob[0], argmax_prob[1], entity_id] = 1
    max_av0_prob = torch.max(prob[..., -2])
    max_av1_prob = torch.max(prob[..., -1])
    if max_av0_prob > max_av1_prob:
        argmax_av0_prob = torch.nonzero(prob[..., -2] == max_av0_prob)[0]
        multilabel[argmax_av0_prob[0], argmax_av0_prob[1], 0] = 0
        multilabel[argmax_av0_prob[0], argmax_av0_prob[1], -2] = 1
    else:
        argmax_av1_prob = torch.nonzero(prob[..., -1] == max_av1_prob)[0]
        multilabel[argmax_av1_prob[0], argmax_av1_prob[1], 0] = 0
        multilabel[argmax_av1_prob[0], argmax_av1_prob[1], -1] = 1
    return multilabel