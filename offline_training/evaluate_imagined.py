'''
Script for evaluating world_model imagined rollouts on stage 2.
'''

import os
import sys

sys.path.append('..')

import argparse
import json
import pickle
import random
import pprint
from collections import defaultdict

from messenger.models.utils import BatchedEncoder
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
import math

from offline_training.batched_world_model.model_new import WorldModel
from chatgpt_groundings.utils import ENTITY_GROUNDING_LOOKUP, MOVEMENT_GROUNDING_LOOKUP, ROLE_GROUNDING_LOOKUP
from dataloader import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm

def encode_manuals(args, manuals, manuals_encoder):
    if args.manuals == 'embed':
        embedded_manuals, _ = manuals_encoder.encode(manuals)
        return embedded_manuals
    return None

def get_parsed_manuals(args, manuals, true_parsed_manuals, gpt_groundings):
    if args.manuals == 'gpt':
        parsed_manuals = [[gpt_groundings[e] for e in manual] for manual in manuals]
    elif args.manuals == 'oracle':
        parsed_manuals = true_parsed_manuals
    else:
        parsed_manuals = None
    return parsed_manuals

def add_metrics(metrics, preds, targets, timesteps):
    # metrics['loc_loss'].extend(
    #     F.cross_entropy(
    #         (preds['loc'].flatten(0, 1) + 1e-6).log(),
    #         targets['loc'].flatten(0, 1),
    #         reduction='none'
    #     ).view(-1).tolist()
    # )

    for i, t in enumerate(timesteps):
        metrics['loc_loss_len_%d' % t].extend(
            F.cross_entropy(
                (preds['loc'][i] + 1e-6).log(),
                targets['loc'][i],
                reduction='none'
            ).view(-1).tolist()
        )
        # for tt in range(1, t + 1):
        #     metrics['loc_loss_len_upto_%d' % t].extend(
        #         F.cross_entropy(
        #             (preds['loc'][i] + 1e-6).log(),
        #             targets['loc'][i],
        #             reduction='none'
        #         ).view(-1).tolist()
        #     )

    # metrics['id_loss'].extend(
    #     F.cross_entropy(
    #         (preds['id'].flatten(0, 1) + 1e-6).log(),
    #         targets['id'].flatten(),
    #         ignore_index=-1,
    #         reduction='none'
    #     ).view(-1).tolist()
    # )

    # for i, t in enumerate(timesteps):
    #     metrics['id_loss_len_%d' % t].extend(
    #         F.cross_entropy(
    #             (preds['id'][i] + 1e-6).log(),
    #             targets['id'][i],
    #             ignore_index=-1,
    #             reduction='none'
    #         ).view(-1).tolist()
    #     )
        # for tt in range(1, t + 1):
        #     metrics['id_loss_len_upto_%d' % t].extend(
        #         F.cross_entropy(
        #             (preds['id'][i] + 1e-6).log(),
        #             targets['id'][i],
        #             ignore_index=-1,
        #             reduction='none'
        #         ).view(-1).tolist()
        #     )

    # metrics['reward_loss'].extend(
    #     F.mse_loss(
    #         preds['reward'],
    #         targets['reward'],
    #         reduction='none'
    #     ).view(-1).tolist()
    # )

    # metrics['done_loss'].extend(
    #     F.binary_cross_entropy(
    #         preds['done'],
    #         targets['done'],
    #         reduction='none'
    #     ).view(-1).tolist()
    # )

def evaluate(args):

    # load world model
    args.learning_rate = 0
    args.weight_decay = 0
    args.reward_loss_weight = 0
    args.done_loss_weight = 0
    args.loss_weights = {
        'loc': 0,
        'id': 0,
        'reward': 0,
        'done': 0
    }
    world_model = WorldModel(args).to(args.device)
    world_model.load_state_dict(torch.load(args.load_model_from, map_location=args.device))
    world_model.eval()

    # Text Encoder
    manuals_encoder_model = AutoModel.from_pretrained("bert-base-uncased")
    manuals_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    manuals_encoder = BatchedEncoder(
        model=manuals_encoder_model,
        tokenizer=manuals_tokenizer,
        device=args.device,
        max_length=36
    )

    # chatgpt groundings
    gpt_groundings = None
    if args.manuals == "gpt":
        with open(args.gpt_groundings_path, "r") as f:
            gpt_groundings = json.load(f)
            # convert groundings into keywords
            for e, grounding in gpt_groundings.items():
                gpt_groundings[e] = [ENTITY_GROUNDING_LOOKUP[grounding[0]], MOVEMENT_GROUNDING_LOOKUP[grounding[1]], ROLE_GROUNDING_LOOKUP[grounding[2]]]

    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # list of test splits
    splits = [split for split in list(dataset["rollouts"].keys()) if "test" in split]
    
    # create dataloaders for each test split in the dataset
    dataloaders = {}
    for split in splits:
        dataloaders[split] = DataLoader(
            dataset,
            split,
            args.max_rollout_length,
            mode="static",
            start_state="initial",
            batch_size=args.batch_size,
            max_rollouts=int(1e8)
        )

    for split, dataloader in dataloaders.items():
        print('evaluating', split)
        metrics = defaultdict(list)
        with torch.no_grad():
            
            manuals, true_parsed_manuals, grids, n_rollouts = dataloader.reset()
            embedded_manuals = encode_manuals(args, manuals, manuals_encoder)
            tensor_grids = torch.from_numpy(grids).long().to(args.device)
            world_model.state_reset(tensor_grids)

            pbar = tqdm(total=n_rollouts)
            while True:
                old_tensor_grids = tensor_grids
                (manuals, true_parsed_manuals, actions, grids, rewards, dones, (new_idxs, cur_idxs),
                timesteps, just_completes, all_complete) = dataloader.step()

                if all_complete:
                    break

                embedded_manuals = encode_manuals(args, manuals, manuals_encoder)

                tensor_actions = torch.from_numpy(actions).long().to(args.device)
                tensor_grids = torch.from_numpy(grids).long().to(args.device)
                tensor_rewards = torch.from_numpy(rewards).float().to(args.device)
                tensor_dones = torch.from_numpy(dones).long().to(args.device)
                tensor_timesteps = torch.from_numpy(timesteps).long().to(args.device)

                if cur_idxs.tolist():

                    parsed_manuals = get_parsed_manuals(args, manuals, true_parsed_manuals, gpt_groundings)

                    preds, targets = world_model.step(
                        old_tensor_grids,
                        embedded_manuals,
                        parsed_manuals,
                        true_parsed_manuals,
                        tensor_actions,
                        tensor_grids,
                        tensor_rewards,
                        tensor_dones,
                        cur_idxs
                    )

                    add_metrics(metrics, preds, targets, timesteps[cur_idxs])

                world_model.state_reset(tensor_grids, new_idxs)

                # change tensor_grids to predicted tensor grids, for imagined rollout
                tensor_grids[cur_idxs] = preds['grid']
                

                pbar.update(just_completes.sum())
            pbar.close()
        
        # save avg loc_perp over steps
        avg_metric = {}
        for k in metrics:
            avg_metric[k] = np.average(metrics[k])
            if 'reward' not in k:
                avg_metric[k.replace('loss', 'perp')] = math.exp(avg_metric[k])
        loc_ces = [avg_metric["loc_loss_len_%d" % t] for t in range(1, args.max_rollout_length)]
        with open(os.path.join(args.output_folder, f'{split}_loc_ces.json'), 'w') as f:
            json.dump(loc_ces, f)
        loc_perps = [avg_metric["loc_perp_len_%d" % t] for t in range(1, args.max_rollout_length)]
        with open(os.path.join(args.output_folder, f'{split}_loc_perps.json'), 'w') as f:
            json.dump(loc_perps, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output_folder", default=None, type=str, help="output folder")
    parser.add_argument("--seed", default=0, type=int, help="Set the seed for the model and evaluation.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")

    # World model arguments
    parser.add_argument("--manuals", default="gpt", type=str,
        choices=['none', 'embed', 'gpt', 'oracle'], help="which type of manuals to pass to the model")
    parser.add_argument("--gpt_groundings_path", default="chatgpt_groundings/chatgpt_grounding_for_text_all.json", type=str, help="path to chatgpt groundings")
    parser.add_argument("--load_model_from", default="experiments/gpt_shuffle_balanced_intentions_10k_train_500_eval/dev_ne_nr_or_nm_best_loc_loss.ckpt", help="Path to world model state dict.")
    parser.add_argument("--hidden_size", default=512, type=int, help="World model hidden size.")
    parser.add_argument('--attr_embed_dim', type=int, default=256, help='attribute embedding size')
    parser.add_argument('--action_embed_dim', type=int, default=256, help='action embedding size')
    parser.add_argument('--desc_key_dim', type=int, default=256, help="description key size")
    parser.add_argument('--keep_entity_features_for_parsed_manuals', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)

    # Dataset arguments
    parser.add_argument("--dataset_path", default="custom_dataset/dataset_shuffle_balanced_intentions_10k_train_500_eval.pickle", help="path to the dataset file")

    # Evaluation arguments
    parser.add_argument("--max_rollout_length", default=32, type=int, help="Max length of a rollout to evaluate for")

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device}")
    if args.output_folder is None:
        args.output_folder = f"evaluation/imagined/{os.path.basename(args.load_model_from).split('.')[0]}/{args.manuals}/"
    os.makedirs(args.output_folder)

    print(pprint.pformat(vars(args), indent=2))

    # seed everything
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_printoptions(precision=1, sci_mode=False, linewidth=100)

    evaluate(args)
