import torch
import torch.nn.functional as F

from messenger.envs.config import NPCS
ENTITY_IDS = {entity.name: entity.id for entity in NPCS}
MOVEMENT_TYPES = {
    "chaser": 0,
    "fleeing": 1,
    "immovable": 2,
}

def batched_convert_grid_to_multilabel(grids):
    multilabels = torch.sum(F.one_hot(grids, num_classes=17), dim=-2)
    multilabels[..., 0] = torch.sum(grids, dim=-1) < 1
    return multilabels

def batched_ground(manuals, ground_truths, world_model):
    query = world_model.sprite_emb(torch.arange(17, device=world_model.device)) # 17 x key_dim
    if world_model.key_type == "oracle":
        keys = F.one_hot(torch.tensor([[ENTITY_IDS[truth[0]] for truth in ground_truth] for ground_truth in ground_truths], device=world_model.device), num_classes=17).float()
    elif "emma" in world_model.key_type:
        # Attention-based text representation        
        keys = world_model.txt_key(manuals)                                     # B x n_sent x sent_len x key_dim
        key_scales = world_model.scale_key(manuals)                             # B x n_sent x sent_len x 1
        keys = keys * key_scales                                                # B x n_sent x sent_len x key_dim
        keys = torch.sum(keys, dim=-2)                                          # B x n_sent x key_dim
    else:
        raise NotImplementedError
        
    kqs = torch.matmul(keys, query.t()).permute(0, 2, 1)                        # B x 17 x n_sent
    if world_model.key_type == "oracle":
        return kqs
    elif "emma" in world_model.key_type:
        masks = (kqs != 0) # keep zeroed-out entries zero
        kqs = kqs / world_model.attn_scale # scale to prevent vanishing grads
        weights = F.softmax(kqs, dim=-1) * masks                                # B x 17 x n_sent
    else:
        # for other key_types, should I use world_model.attn_scale?
        raise NotImplementedError
    
    return weights

def batched_convert_multilabel_to_emb(multilabels, manuals, ground_truths, world_model):
    if world_model.val_type == "oracle":
        values = F.one_hot(torch.tensor([[MOVEMENT_TYPES[truth[1]] for truth in ground_truth] for ground_truth in ground_truths], device=world_model.device), num_classes=4)
    elif "emma" in world_model.val_type:
        values = world_model.txt_val(manuals)                                        # B x n_sent x sent_len x val_dim
        val_scales = world_model.scale_val(manuals)                                  # B x n_sent x sent_len x 1
        values = values * val_scales                                                 # B x n_sent x sent_len x val_dim
        values = torch.sum(values, dim=-2)                                           # B x n_sent x val_dim
    else:
        raise NotImplementedError

    weights = batched_ground(manuals, ground_truths, world_model)                    # B x 17 x n_sent
    entity_values = torch.sum(weights.unsqueeze(-1) * values.unsqueeze(-3), dim=-2) # B x 17 x val_dim
    entity_values[:, 0] = torch.tensor([0], device=world_model.device)
    entity_values[:, 1] = torch.tensor([0], device=world_model.device)
    entity_values[:, 14] = torch.tensor([0], device=world_model.device)
    entity_values[:, 15] = world_model.avatar_no_message_val_emb
    entity_values[:, 16] = world_model.avatar_with_message_val_emb
    entity_values = entity_values.unsqueeze(-3).unsqueeze(-3)*multilabels[..., None] # B x 10 x 10 x 17 x val_dim
    entity_values = torch.sum(entity_values, dim=-2)                                 # B x 10 x 10 x val_dim
    return torch.cat((multilabels[..., world_model.relevant_cls_idxs], entity_values), dim=-1)

def batched_convert_prob_to_multilabel(probs, nonexistence_probs, prediction_type, threshold, refine, entity_ids):
    if prediction_type == "existence":
        multilabels = 1*(probs > threshold) # B x 10 x 10 x 17
    elif prediction_type == "class":
        multilabels = 1*(probs > probs[..., 0:1])
    elif prediction_type == "location":
        multilabels = 1*(probs > nonexistence_probs[:, None, None, :])
    if refine:
        multilabels = multilabels*(probs >= torch.amax(probs, dim=(1, 2), keepdim=True))
        multilabels[..., 15:17] = (probs[..., 15:17] >= torch.amax(probs[..., 15:17], dim=(1, 2, 3), keepdim=True))
        multilabels[..., :15] = multilabels[..., :15]*(F.one_hot(entity_ids, num_classes=15).sum(dim=-2))[:, None, None, :]
    multilabels[..., 0] = (multilabels[..., 1:].sum(dim=-1) < 1)
    return multilabels