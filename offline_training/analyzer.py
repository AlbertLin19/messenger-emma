import torch
import torch.nn.functional as F
import wandb

from matplotlib import pyplot as plt
from offline_training.utils import batched_ground, ENTITY_IDS

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
    def __init__(self, world_model, log_prefix, max_rollout_length, relevant_cls_idxs, n_frames, device):
        self.world_model = world_model
        self.log_prefix = log_prefix
        self.max_rollout_length = max_rollout_length
        self.relevant_cls_idxs = relevant_cls_idxs.cpu()
        self.n_frames = n_frames
        self.device = device

        self.sprite_idxs = self.relevant_cls_idxs[((self.relevant_cls_idxs != 0)*(self.relevant_cls_idxs != 1)*(self.relevant_cls_idxs != 14)).argwhere().squeeze(-1)]
        self.entity_idxs = self.sprite_idxs[((self.sprite_idxs != 15)*(self.sprite_idxs != 16)).argwhere().squeeze(-1)]

    def reset(self):
        self.tps = torch.zeros((self.max_rollout_length, 17), dtype=int, device=self.device)
        self.fns = torch.zeros((self.max_rollout_length, 17), dtype=int, device=self.device)
        self.fps = torch.zeros((self.max_rollout_length, 17), dtype=int, device=self.device)

        self.pred_probs_for_vid = []
        self.pred_multilabels_for_vid = []
        self.true_probs_for_vid = []
        self.true_multilabels_for_vid = []

        self.manuals = {name: None for name in ENTITY_IDS.keys()}
        self.ground_truths = {name: None for name in ENTITY_IDS.keys()}

    def push(self, pred_probs_tuple, pred_multilabels, true_probs, true_multilabels, descriptors_tuple, ground_truths, idxs_tuple, timesteps, step):
        pred_probs, pred_nonexistence_probs = pred_probs_tuple
        manuals, tokens = descriptors_tuple
        new_idxs, cur_idxs = idxs_tuple

        with torch.no_grad():
            # increment tps, fns, and fps for ongoing trajectories
            confusions = pred_multilabels / true_multilabels # 1 -> tp, 0 -> fn, inf -> fp, nan -> tn
            self.tps.scatter_add_(dim=0, index=timesteps[cur_idxs].unsqueeze(-1).expand(-1, 17), src=torch.sum(confusions == 1, dim=(1, 2))[cur_idxs])
            self.fns.scatter_add_(dim=0, index=timesteps[cur_idxs].unsqueeze(-1).expand(-1, 17), src=torch.sum(confusions == 0, dim=(1, 2))[cur_idxs])
            self.fps.scatter_add_(dim=0, index=timesteps[cur_idxs].unsqueeze(-1).expand(-1, 17), src=torch.sum(confusions == float('inf'), dim=(1, 2))[cur_idxs])

            # store single frame from first trajectory in batch (if ongoing) for each of the videos
            ongoing = (cur_idxs == 0).any()
            self.pred_probs_for_vid.append(pred_probs[0].cpu()*(1 if ongoing else 0))
            if len(self.pred_probs_for_vid) > self.n_frames:
                self.pred_probs_for_vid.pop(0)
            self.pred_multilabels_for_vid.append(pred_multilabels[0].cpu()*(1 if ongoing else 0))
            if len(self.pred_multilabels_for_vid) > self.n_frames:
                self.pred_multilabels_for_vid.pop(0)
            self.true_probs_for_vid.append(true_probs[0].cpu()*(1 if ongoing else 0))
            if len(self.true_probs_for_vid) > self.n_frames:
                self.true_probs_for_vid.pop(0)
            self.true_multilabels_for_vid.append(true_multilabels[0].cpu()*(1 if ongoing else 0))
            if len(self.true_multilabels_for_vid) > self.n_frames:
                self.true_multilabels_for_vid.pop(0)

            # accumulate manuals and ground_truths
            missing = None in self.manuals.values()
            for i in range(len(manuals)):
                if not missing:
                    break

                for j in range(len(manuals[i])):
                    name = ground_truths[i][j][0]

                    if self.manuals[name] is None:
                        self.manuals[name] = manuals[i][j]
                        self.ground_truths[name] = ground_truths[i][j]

                        missing = None in self.manuals.values()
                        if not missing:
                            break

    def getLog(self):
        log = {}

        # calculate recalls, precisions, f1s
        recalls = self.tps / (self.tps + self.fns)
        precisions = self.tps / (self.tps + self.fps)
        f1s = (2 * recalls * precisions) / (recalls + precisions)

        # log recall, precision, f1 versus timesteps
        def logCurves(log_suffix, idxs):
            recall_fig, recall_ax = plt.subplots()
            recall_ax.plot(np.arange(self.max_rollout_length), recalls[:, idxs].mean(dim=-1).cpu().numpy())

            precision_fig, precision_ax = plt.subplots()
            precision_ax.plot(np.arange(self.max_rollout_length), precisions[:, idxs].mean(dim=-1).cpu().numpy())

            f1_fig, f1_ax = plt.subplots()
            f1_ax.plot(np.arange(self.max_rollout_length), f1s[:, idxs].mean(dim=-1).cpu().numpy())
            
            log.update({
                f'recall_{log_suffix}': recall_fig,
                f'precision_{log_suffix}': precision_fig,
                f'f1_{log_suffix}': f1_fig,
            })

        for idx in self.relevant_cls_idxs:
            logCurves(idx, [idx])
        logCurves('sprite', self.sprite_idxs)
        logCurves('entity', self.entity_idxs)
        logCurves('avatar', [15, 16])
        
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
