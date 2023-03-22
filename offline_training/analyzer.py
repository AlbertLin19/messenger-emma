import torch
import torch.nn.functional as F
import numpy as np
import wandb

from offline_training.batched_world_model.utils import batched_ground, ENTITY_IDS

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
        self.sorted_entity_idxs = torch.sort(self.entity_idxs).values

        self.reset()

    def reset(self):
        self.tps = torch.zeros((self.max_rollout_length, 17), dtype=int, device=self.device)
        self.fns = torch.zeros((self.max_rollout_length, 17), dtype=int, device=self.device)
        self.fps = torch.zeros((self.max_rollout_length, 17), dtype=int, device=self.device)

        self.pred_probs_for_vid = []
        self.pred_multilabels_for_vid = []
        self.true_probs_for_vid = []
        self.true_multilabels_for_vid = []

        self.manual = {idx.item(): None for idx in self.entity_idxs}
        self.ground_truth = {idx.item(): None for idx in self.entity_idxs}
        self.token = {idx.item(): None for idx in self.entity_idxs}

        self.game_grounding = torch.zeros((len(self.entity_idxs), len(self.entity_idxs)), device=self.device)

        self.ln_perplexities = torch.zeros(17, dtype=float, device=self.device)
        self.ln_perplexity_counts = torch.zeros(17, dtype=int, device=self.device)
        self.nontrivial_ln_perplexities = torch.zeros(17, dtype=float, device=self.device)
        self.nontrivial_ln_perplexity_counts = torch.zeros(17, dtype=int, device=self.device)

    def push(self, pred_probs_tuple, pred_multilabels, true_probs, true_multilabels, descriptors_tuple, ground_truths, idxs_tuple, entity_ids, timesteps):
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
            if (cur_idxs == 0).any():
                self.pred_probs_for_vid.append(pred_probs[0].cpu())
                if len(self.pred_probs_for_vid) > self.n_frames:
                    self.pred_probs_for_vid.pop(0)
                self.pred_multilabels_for_vid.append(pred_multilabels[0].cpu())
                if len(self.pred_multilabels_for_vid) > self.n_frames:
                    self.pred_multilabels_for_vid.pop(0)
                self.true_probs_for_vid.append(true_probs[0].cpu())
                if len(self.true_probs_for_vid) > self.n_frames:
                    self.true_probs_for_vid.pop(0)
                self.true_multilabels_for_vid.append(true_multilabels[0].cpu())
                if len(self.true_multilabels_for_vid) > self.n_frames:
                    self.true_multilabels_for_vid.pop(0)

            # accumulate manuals and ground_truths
            missing = None in self.manual.values()
            for i in new_idxs:
                if not missing:
                    break

                for j in range(len(manuals[i])):
                    idx = ENTITY_IDS[ground_truths[i][j][0]]

                    if self.manual[idx] is None:
                        self.manual[idx] = manuals[i][j]
                        self.ground_truth[idx] = ground_truths[i][j]
                        self.token[idx] = tokens[i][j]

                        missing = None in self.manual.values()
                        if not missing:
                            break

            # accumulate game_grounding as an alternative to the full-sample grounding
            missing = (self.game_grounding.sum(dim=-1) == 0).any()
            for i in new_idxs:
                if not missing:
                    break

                for j in range(len(manuals[i])):
                    idx = (self.sorted_entity_idxs == ENTITY_IDS[ground_truths[i][j][0]]).argwhere().squeeze()

                    if self.game_grounding[idx].sum() == 0:
                        self.game_grounding[idx, torch.cat([(self.sorted_entity_idxs == ENTITY_IDS[ground_truths[i][k][0]]).argwhere().squeeze(0) for k in range(len(manuals[i]))])] = batched_ground(manuals[i].unsqueeze(0), [ground_truths[i]], self.world_model)[0, ENTITY_IDS[ground_truths[i][j][0]]]

                        missing = (self.game_grounding.sum(dim=-1) == 0).any()
                        if not missing:
                            break

            # accumulate ln_perplexities
            all_pred_probs = torch.cat((pred_probs.permute(0, 3, 1, 2).flatten(2, 3), pred_nonexistence_probs.unsqueeze(-1)), dim=-1) # B x 17 x 101
            all_pred_log_probs = torch.log(all_pred_probs)
            true_nonexistence_probs = 1.0*(torch.sum(true_probs, dim=(1, 2)) <= 0)
            all_true_probs = torch.cat((true_probs.permute(0, 3, 1, 2).flatten(2, 3), true_nonexistence_probs.unsqueeze(-1)), dim=-1)
            self.ln_perplexities -= torch.sum((all_true_probs*all_pred_log_probs)[cur_idxs], dim=(0, 2))
            self.ln_perplexity_counts += len(cur_idxs)

            # accumulate nontrivial_ln_perplexities
            # using entity_ids: B x 3 array, where each row holds the 3 entity ids of a game
            entity_masks = torch.sum(F.one_hot(entity_ids, num_classes=17), dim=1).unsqueeze(-1) # B x 17 x 1
            entity_masks[:, 15:17] = 1 # avatar with/without message is always possible in a game
            self.nontrivial_ln_perplexities -= torch.sum((entity_masks*all_true_probs*all_pred_log_probs)[cur_idxs], dim=(0, 2))
            self.nontrivial_ln_perplexity_counts += torch.sum(entity_masks[cur_idxs], dim=0).squeeze(-1)

    def getLog(self, step):
        log = {}

        with torch.no_grad():
            # calculate recalls, precisions, f1s
            recalls = self.tps / (self.tps + self.fns)
            precisions = self.tps / (self.tps + self.fps)
            f1s = (2 * recalls * precisions) / (recalls + precisions)

            # log recall, precision, f1 for each timestep and their averages across timesteps
            def logCurves(log_suffix, idxs):
                recall = recalls[:, idxs].nanmean(dim=-1)
                precision = precisions[:, idxs].nanmean(dim=-1)
                f1 = f1s[:, idxs].nanmean(dim=-1)
                        
                timestep = np.arange(self.max_rollout_length)
                log.update({
                    f'recall_{log_suffix}': wandb.plot.line(wandb.Table(data=np.stack((timestep, recall.cpu().numpy()), axis=-1), columns=['timestep', 'recall']), 'timestep', 'recall', title=f'{log_suffix}: Recall versus Timestep'),
                    f'precision_{log_suffix}': wandb.plot.line(wandb.Table(data=np.stack((timestep, precision.cpu().numpy()), axis=-1), columns=['timestep', 'precision']), 'timestep', 'precision', title=f'{log_suffix}: Precision versus Timestep'),
                    f'f1_{log_suffix}': wandb.plot.line(wandb.Table(data=np.stack((timestep, f1.cpu().numpy()), axis=-1), columns=['timestep', 'f1']), 'timestep', 'f1', title=f'{log_suffix}: F1 versus Timestep'),
                    f'recall_{log_suffix}_avg': recall.nanmean(),
                    f'precision_{log_suffix}_avg': precision.nanmean(),
                    f'f1_{log_suffix}_avg': f1.nanmean(),
                })
                log.update({f'recall_{log_suffix}_{i}': recall[i] for i in range(self.max_rollout_length)})
                log.update({f'precision_{log_suffix}_{i}': precision[i] for i in range(self.max_rollout_length)})
                log.update({f'f1_{log_suffix}_{i}': f1[i] for i in range(self.max_rollout_length)})

            for idx in self.relevant_cls_idxs:
                logCurves(idx, [idx])
            logCurves('sprite', self.sprite_idxs)
            logCurves('entity', self.entity_idxs)
            logCurves('avatar', [15, 16])

            # log visualizations of predictions
            
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

            # log grounding and token attention table
            if not (None in self.manual.values()):
                manual = torch.stack([self.manual[idx.item()] for idx in self.sorted_entity_idxs], dim=0)
                ground_truth = [self.ground_truth[idx.item()] for idx in self.sorted_entity_idxs]
                token = [self.token[idx.item()] for idx in self.sorted_entity_idxs]

                grounding = batched_ground(manual.unsqueeze(0), [ground_truth], self.world_model)[0, self.sorted_entity_idxs].cpu()
                log.update({'grounding': wandb.Image(grounding.unsqueeze(0))})

                if ('emma' in self.world_model.key_type) or ('emma' in self.world_model.val_type):
                    token = np.asarray(token)
                    columns = [step*np.ones(token.size), token.flatten()]
                    column_names = ['step', 'token']
                    if 'emma' in self.world_model.key_type:
                        key_attention = self.world_model.scale_key(manual).squeeze(-1).cpu()
                        columns.append(key_attention.numpy().flatten())
                        column_names.append('key')
                    if 'emma' in self.world_model.val_type:
                        value_attention = self.world_model.scale_val(manual).squeeze(-1).cpu()
                        columns.append(value_attention.numpy().flatten())
                        column_names.append('value')
                    log.update({'token_attention': wandb.Table(columns=column_names, data=np.stack(columns, axis=-1))})

            # log game_grounding as an alternative
            if not (self.game_grounding.sum(dim=-1) == 0).any():
                log.update({'game_grounding': wandb.Image(self.game_grounding.cpu().unsqueeze(0))})

            # log perplexities
            log.update({f'perplexity_{i}': np.exp((self.ln_perplexities[i] / self.ln_perplexity_counts[i]).cpu().numpy()) for i in self.relevant_cls_idxs})
            log.update({f'nontrivial_perplexity_{i}': np.exp((self.nontrivial_ln_perplexities[i] / self.nontrivial_ln_perplexity_counts[i]).cpu().numpy()) for i in self.relevant_cls_idxs})
                        
        for key in list(log.keys()):
            log[self.log_prefix + key] = log.pop(key)
        return log
