import torch
import torch.nn.functional as F
import wandb

COLORS = torch.tensor([
        [0, 0, 0], # 0 background
        [255, 0, 0], # 1 dirt
        [255, 85, 0], # 2 airplane
        [255, 170, 0], # 3 mage
        [255, 255, 0], # 4 dog
        [170, 255, 0], # 5 bird
        [85, 255, 0], # 6 fish
        [0, 255, 0], # 7 scientist
        [0, 255, 85], # 8 thief
        [0, 255, 170], # 9 ship
        [0, 255, 255], # 10 ball
        [0, 170, 255], # 11 robot
        [0, 85, 255], # 12 queen
        [0, 0, 255], # 13 sword
        [85, 0, 255], # 14 wall
        [170, 0, 255], # 15 no_message
        [255, 0, 255], # 16 with_message
    ])

    

class Analyzer:
    def __init__(self, log_prefix, eval_length, vis_length, relevant_cls_idxs):
        self.log_prefix = log_prefix
        self.eval_length = eval_length
        self.vis_length = vis_length
        self.relevant_cls_idxs = relevant_cls_idxs.cpu()

        self.pred_probs_for_vid = []
        self.pred_multilabels_for_vid = []
        self.true_probs_for_vid = []
        self.true_multilabels_for_vid = []

        self.tps = []
        self.fns = []
        self.fps = []

    def push(self, pred_probs_tuple, pred_multilabels, true_probs, true_multilabels):
        pred_probs, pred_nonexistence_probs = pred_probs_tuple
        with torch.no_grad():
            # calculate confusions
            confusions = pred_multilabels / true_multilabels # 1 -> tp, 0 -> fn, inf -> fp, nan -> tn
            self.tps.append(torch.sum(confusions == 1, dim=(0, 1, 2)))            
            if len(self.tps) > self.eval_length:
                self.tps.pop(0)
            self.fns.append(torch.sum(confusions == 0, dim=(0, 1, 2)))
            if len(self.fns) > self.eval_length:
                self.fns.pop(0)
            self.fps.append(torch.sum(confusions == float('inf'), dim=(0, 1, 2)))
            if len(self.fps) > self.eval_length:
                self.fps.pop(0)

            # store single frame for each videos
            self.pred_probs_for_vid.append(pred_probs[0].cpu())
            if len(self.pred_probs_for_vid) > self.vis_length:
                self.pred_probs_for_vid.pop(0)
            self.pred_multilabels_for_vid.append(pred_multilabels[0].cpu())
            if len(self.pred_multilabels_for_vid) > self.vis_length:
                self.pred_multilabels_for_vid.pop(0)
            self.true_probs_for_vid.append(true_probs[0].cpu())
            if len(self.true_probs_for_vid) > self.vis_length:
                self.true_probs_for_vid.pop(0)
            self.true_multilabels_for_vid.append(true_multilabels[0].cpu())
            if len(self.true_multilabels_for_vid) > self.vis_length:
                self.true_multilabels_for_vid.pop(0)

    def getLog(self):
        log = {}

        # calculate recall, precision, f1
        tp = torch.stack(self.tps, dim=0).sum(dim=0)
        fn = torch.stack(self.fns, dim=0).sum(dim=0)
        fp = torch.stack(self.fps, dim=0).sum(dim=0)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = (2 * recall * precision) / (recall + precision)
        sprite_idxs = self.relevant_cls_idxs[((self.relevant_cls_idxs != 0)*(self.relevant_cls_idxs != 1)*(self.relevant_cls_idxs != 14)).argwhere().squeeze(-1)]
        entity_idxs = sprite_idxs[((sprite_idxs != 15)*(sprite_idxs != 16)).argwhere().squeeze(-1)]
        log.update({
            'recall_sprite': recall[sprite_idxs].nanmean(),
            'precision_sprite': precision[sprite_idxs].nanmean(),
            'f1_sprite': f1[sprite_idxs].nanmean(),
            'recall_entity': recall[entity_idxs].nanmean(),
            'precision_entity': precision[entity_idxs].nanmean(),
            'f1_entity': f1[entity_idxs].nanmean(),
            'recall_avatar': recall[15:17].nanmean(),
            'precision_avatar': precision[15:17].nanmean(),
            'f1_avatar': f1[15:17].nanmean(),
        })
        log.update({f'recall_{i}': recall[i] for i in self.relevant_cls_idxs})
        log.update({f'precision_{i}': precision[i] for i in self.relevant_cls_idxs})
        log.update({f'f1_{i}': f1[i] for i in self.relevant_cls_idxs})
        
        true_probs = F.pad(torch.stack(self.true_probs_for_vid, dim=0), (0, 0, 1, 1, 1, 1))
        pred_probs = F.pad(torch.stack(self.pred_probs_for_vid, dim=0), (0, 0, 1, 1, 1, 1))
        probs = torch.cat((true_probs, pred_probs), dim=2)
        log.update({f'prob_{i}': wandb.Video((255*probs[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in self.relevant_cls_idxs})
        log.update({'probs': wandb.Video(torch.sum((probs.unsqueeze(-1)*COLORS)[..., self.relevant_cls_idxs, :], dim=-2).permute(0, 3, 1, 2).to(torch.uint8))})

        true_multilabels = F.pad(torch.stack(self.true_multilabels_for_vid, dim=0), (0, 0, 1, 1, 1, 1))
        pred_multilabels = F.pad(torch.stack(self.pred_multilabels_for_vid, dim=0), (0, 0, 1, 1, 1, 1))
        multilabels = torch.cat((true_multilabels, pred_multilabels), dim=2)
        log.update({f'multilabel_{i}': wandb.Video((255*multilabels[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in self.relevant_cls_idxs})
        log.update({'multilabels': wandb.Video(torch.min(torch.sum((multilabels.unsqueeze(-1)*COLORS)[..., self.relevant_cls_idxs, :], dim=-2), torch.tensor([255])).permute(0, 3, 1, 2).to(torch.uint8))})

        for key in list(log.keys()):
            log[self.log_prefix + key] = log.pop(key)
        return log
