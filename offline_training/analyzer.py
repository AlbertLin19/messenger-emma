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
    def __init__(self, log_prefix, eval_length, vis_length):
        self.log_prefix = log_prefix
        self.eval_length = eval_length
        self.vis_length = vis_length

        self.pred_probs_for_vid = []
        self.pred_multilabels_for_vid = []
        self.true_probs_for_vid = []
        self.true_multilabels_for_vid = []

        # self.real_step_count = 0
        # self.imag_step_count = 0
        # self.real_loss_total = 0
        # self.imag_loss_total = 0

        # self.real_tp = torch.zeros(17, dtype=int, device=device)
        # self.real_fn = torch.zeros(17, dtype=int, device=device)
        # self.real_fp = torch.zeros(17, dtype=int, device=device)
        # self.real_tn = torch.zeros(17, dtype=int, device=device)

        # self.imag_tp = torch.zeros(17, dtype=int, device=device)
        # self.imag_fn = torch.zeros(17, dtype=int, device=device)
        # self.imag_fp = torch.zeros(17, dtype=int, device=device)
        # self.imag_tn = torch.zeros(17, dtype=int, device=device)

        # self.real_dists = []
        # self.imag_dists = []

    def push(self, pred_probs_tuple, pred_multilabels, true_probs, true_multilabels):
        pred_probs, pred_nonexistence_probs = pred_probs_tuple
        with torch.no_grad():
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

        true_probs = F.pad(torch.stack(self.true_probs_for_vid, dim=0), (0, 0, 1, 1, 1, 1))
        pred_probs = F.pad(torch.stack(self.pred_probs_for_vid, dim=0), (0, 0, 1, 1, 1, 1))
        probs = torch.cat((true_probs, pred_probs), dim=2)
        log.update({f'prob_{i}': wandb.Video((255*probs[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
        log.update({'probs': wandb.Video(torch.sum(probs.unsqueeze(-1)*COLORS, dim=-2).permute(0, 3, 1, 2).to(torch.uint8))})

        true_multilabels = F.pad(torch.stack(self.true_multilabels_for_vid, dim=0), (0, 0, 1, 1, 1, 1))
        pred_multilabels = F.pad(torch.stack(self.pred_multilabels_for_vid, dim=0), (0, 0, 1, 1, 1, 1))
        multilabels = torch.cat((true_multilabels, pred_multilabels), dim=2)
        log.update({f'multilabel_{i}': wandb.Video((255*multilabels[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
        log.update({'multilabels': wandb.Video(torch.min(torch.sum(multilabels.unsqueeze(-1)*COLORS, dim=-2), torch.tensor([255])).permute(0, 3, 1, 2).to(torch.uint8))})

        for key in list(log.keys()):
            log[self.log_prefix + key] = log.pop(key)
        return log
