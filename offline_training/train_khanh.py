'''
Script for offline training world models on Stage 2.
'''
import os
import sys
import json
import argparse
import time
import pickle
import random
import math
import pprint
from collections import defaultdict

import torch
import torch.nn.functional as F
import wandb
import numpy as np


sys.path.append('..')

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from offline_training.batched_world_model.model_khanh import ENTITY_IDS
from messenger.models.utils import BatchedEncoder
from offline_training.batched_world_model.model_khanh import WorldModel
from dataloader import DataLoader
from evaluator import Evaluator

from chatgpt_groundings.utils import ENTITY_GROUNDING_LOOKUP, MOVEMENT_GROUNDING_LOOKUP, ROLE_GROUNDING_LOOKUP

def train(args):

    args.loss_weights = {
        'loc': 1,
        'id': 1,
        'reward': args.reward_loss_weight,
        'done': args.done_loss_weight
    }

    world_model = WorldModel(args).to(args.device)
    print(world_model)

    if args.load_model_from:
        model_state_dict = torch.load(args.load_model_from)
        world_model.load_state_dict(model_state_dict)
        print('Loaded model from', args.load_model_from)

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

    # list of splits
    splits = list(dataset["rollouts"].keys())
    # exclude all test splits
    exclude_split = 'dev' if args.eval_mode else 'test'
    splits = [split for split in splits if exclude_split not in split]
    print('Included splits', splits)
    train_split = None
    for split in splits:
        if "train" in split:
            train_split = split
            break
    assert train_split is not None

    # create dataloaders for each split in the dataset
    train_dataloader = DataLoader(
        dataset,
        train_split,
        args.max_rollout_length,
        mode="random",
        start_state=args.train_start_state,
        batch_size=args.batch_size,
    )

    eval_dataloaders = {}
    for split in splits:
        max_rollouts = int(1e8)
        if 'train' in split:
            max_rollouts = 100
        eval_dataloaders[split] = DataLoader(
            dataset,
            split,
            args.max_rollout_length,
            mode="static",
            start_state="initial",
            batch_size=args.eval_batch_size,
            max_rollouts=max_rollouts
        )

    # training variables
    step = 0
    start_time = time.time()

    # load initial data
    manuals, true_parsed_manuals, grids = train_dataloader.reset()
    embedded_manuals = encode_manuals(args, manuals, manuals_encoder)

    tensor_grids = torch.from_numpy(grids).long().to(args.device)
    tensor_rewards = torch.zeros(tensor_grids.shape[0]).float().to(args.device)

    # reset world_model hidden states
    world_model.state_reset(tensor_grids)

    best_metric = defaultdict(lambda: defaultdict(lambda: 1e9))
    train_metrics = defaultdict(list)

    while step < args.max_step:

        # EVALUATION
        if step % args.eval_step == 0:
            wandb_stats = {}
            wandb_stats['step'] = step
            log_str = []
            avg_train_metric = {}
            for k in train_metrics:
                avg_train_metric[k] = np.average(train_metrics[k])
                log_str.append('%s %.4f' % (k, avg_train_metric[k]))
                wandb_stats['train/' + k] = avg_train_metric[k]
            log_str = ', '.join(log_str)
            print()
            print('After %d step' % step)
            print('  TRAIN', log_str)
            print()

            # reset train metrics
            train_metrics = defaultdict(list)

            for eval_split, eval_dataloader in eval_dataloaders.items():

                eval_world_model = WorldModel(args).to(args.device)
                eval_world_model.load_state_dict(world_model.state_dict())

                with torch.no_grad():
                    eval_metric = evaluate(
                        args,
                        eval_split,
                        step,
                        eval_world_model,
                        gpt_groundings,
                        manuals_encoder,
                        eval_dataloader,
                        best_metric[eval_split]
                    )
                    for k in eval_metric:
                        wandb_stats[('%s/' % eval_split) + k] = eval_metric[k]
                        wandb_stats[('%s_best/' % eval_split) + k] = best_metric[eval_split][k]

            if args.use_wandb:
                wandb.log(wandb_stats)

        if args.eval_mode:
            break

        # TRAIN
        world_model.train()

        old_tensor_grids = tensor_grids
        old_tensor_rewards = tensor_rewards

        # load next-step data
        manuals, true_parsed_manuals, actions, grids, rewards, dones, (new_idxs, cur_idxs), timesteps = train_dataloader.step()
        embedded_manuals = encode_manuals(args, manuals, manuals_encoder)

        tensor_actions = torch.from_numpy(actions).long().to(args.device)
        tensor_grids = torch.from_numpy(grids).long().to(args.device)
        tensor_rewards = torch.from_numpy(rewards).float().to(args.device)
        tensor_dones = torch.from_numpy(dones).long().to(args.device)
        tensor_timesteps = torch.from_numpy(timesteps).long().to(args.device)

        parsed_manuals = get_parsed_manuals(args, manuals, true_parsed_manuals, gpt_groundings)

        if cur_idxs.tolist():
            world_model.step(
                old_tensor_grids,
                old_tensor_rewards,
                embedded_manuals,
                parsed_manuals,
                true_parsed_manuals,
                tensor_actions,
                tensor_grids,
                tensor_rewards,
                tensor_dones,
                cur_idxs
            )

        step += 1

        # perform update
        if step % args.update_step == 0:
            avg_loss = world_model.loss_update()
            for k in avg_loss:
                train_metrics[k].append(avg_loss[k])

        # reset world_model hidden states for new rollouts
        world_model.state_reset(tensor_grids, new_idxs)

        # check if max_time has elapsed
        if time.time() - start_time > 60 * 60 * args.max_time:
            break

def evaluate(args, split, step, world_model, gpt_groundings, manuals_encoder, dataloader, best_metric):

    world_model.eval()

    manuals, true_parsed_manuals, grids, n_rollouts = dataloader.reset()
    embedded_manuals = encode_manuals(args, manuals, manuals_encoder)
    tensor_grids = torch.from_numpy(grids).long().to(args.device)
    tensor_rewards = torch.zeros(tensor_grids.shape[0]).float().to(args.device)

    world_model.state_reset(tensor_grids)

    metrics = defaultdict(list)
    while True:

        old_tensor_grids = tensor_grids
        old_tensor_rewards = tensor_rewards

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
                old_tensor_rewards,
                embedded_manuals,
                parsed_manuals,
                true_parsed_manuals,
                tensor_actions,
                tensor_grids,
                tensor_rewards,
                tensor_dones,
                cur_idxs
            )

            add_eval_metrics(metrics, preds, targets, timesteps[cur_idxs])

        world_model.state_reset(tensor_grids, new_idxs)

    avg_metric = {}
    for k in metrics:
        avg_metric[k] = np.average(metrics[k])
        avg_metric[k.replace('loss', 'perp')] = math.exp(avg_metric[k])
    avg_metric['total_loss'] = avg_metric['loc_loss'] + avg_metric['id_loss'] + avg_metric['reward_loss'] + avg_metric['done_loss']

    # update best model
    for k in avg_metric:
        if avg_metric[k] < best_metric[k]:
            best_metric[k] = avg_metric[k]
            if k in ['loc_loss', 'id_loss', 'total_loss']:
                model_path = os.path.join(args.output, '%s_best_%s.ckpt' % (split, k))
                if not args.eval_mode:
                    torch.save(world_model.state_dict(), model_path)
                    print('Saved best %s %s to %s' % (split, k, model_path))

    print('  EVALUATION on %s' % split)
    logged_losses = ['total_loss', 'loc_loss', 'id_loss', 'reward_loss', 'done_loss']
    log_str = []
    for k in logged_losses:
        log_str.append('%s %.4f' % (k, avg_metric[k]))
    log_str = '    CURRENT ' + ', '.join(log_str)
    print(log_str)
    log_str = []
    for k in logged_losses:
        log_str.append('%s %.4f' % (k, best_metric[k]))
    log_str = '    BEST    ' + ', '.join(log_str)
    print(log_str)

    # evaluate grounding
    if args.manuals == 'embed':
        avg_metric['grounding'] = evaluate_grounding(args, world_model, manuals_encoder, dataloader)
    return avg_metric

def evaluate_grounding(args, world_model, manuals_encoder, dataloader):
    entity_ids = list(ENTITY_IDS.values())
    entity_ids.sort()
    embedded_manual = torch.zeros((len(entity_ids), 36, 768), device=args.device)

    # gather one description each
    for i in range(len(entity_ids)):
        found = False
        for j in range(dataloader.n_rollouts):
            if found:
                break
            for k in range(len(dataloader.ground_truths_array[j])):
                if ENTITY_IDS[dataloader.ground_truths_array[j][k][0]] == entity_ids[i]:
                    embedded_manual[i], _ = manuals_encoder.encode([[dataloader.manuals_array[j][k]]])
                    found = True
                    break
        if not found:
            raise RuntimeError

    # compute grounding
    entity_query = world_model.entity_query_embeddings(torch.tensor(entity_ids).to(args.device)) # 12 x key_dim
    desc_key = torch.sum(world_model.token_key_att(embedded_manual)*world_model.token_key(embedded_manual), dim=-2) # 12 x key_dim
    desc_att_logits = torch.mm(entity_query, desc_key.T) # 12 (entities) x 12 (desc) grounding
    desc_att = F.softmax(desc_att_logits / np.sqrt(world_model.desc_key_dim), dim=-1)
    return wandb.Image(desc_att.unsqueeze(0))

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

def add_eval_metrics(metrics, preds, targets, timesteps):

    metrics['loc_loss'].extend(
        F.cross_entropy(
            (preds['loc'].flatten(0, 1) + 1e-6).log(),
            targets['loc'].flatten(0, 1),
            reduction='none'
        ).view(-1).tolist()
    )

    for i, t in enumerate(timesteps):
        if t <= 10:
            metrics['loc_loss_len_%d' % t].extend(
                F.cross_entropy(
                    (preds['loc'][i] + 1e-6).log(),
                    targets['loc'][i],
                    reduction='none'
                ).view(-1).tolist()
            )
            for tt in range(1, t + 1):
                metrics['loc_loss_len_upto_%d' % t].extend(
                    F.cross_entropy(
                        (preds['loc'][i] + 1e-6).log(),
                        targets['loc'][i],
                        reduction='none'
                    ).view(-1).tolist()
                )

    metrics['id_loss'].extend(
        F.cross_entropy(
            (preds['id'].flatten(0, 1) + 1e-6).log(),
            targets['id'].flatten(),
            ignore_index=-1,
            reduction='none'
        ).view(-1).tolist()
    )

    for i, t in enumerate(timesteps):
        if t <= 10:
            metrics['id_loss_len_%d' % t].extend(
                F.cross_entropy(
                    (preds['id'][i] + 1e-6).log(),
                    targets['id'][i],
                    ignore_index=-1,
                    reduction='none'
                ).view(-1).tolist()
            )
            for tt in range(1, t + 1):
                metrics['id_loss_len_upto_%d' % t].extend(
                    F.cross_entropy(
                        (preds['id'][i] + 1e-6).log(),
                        targets['id'][i],
                        ignore_index=-1,
                        reduction='none'
                    ).view(-1).tolist()
                )

    metrics['reward_loss'].extend(
        F.cross_entropy(
            preds['reward'],
            targets['reward'],
            reduction='none'
        ).view(-1).tolist()
    )

    metrics['done_loss'].extend(
        F.binary_cross_entropy(
            preds['done'],
            targets['done'],
            reduction='none'
        ).view(-1).tolist()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output", default=None, type=str, help="Local output file name or path.")
    parser.add_argument("--seed", default=123, type=int, help="Set the seed for the model and training.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--eval_mode", type=int, default=0, help="evaluation mode")

    # text config
    parser.add_argument("--manuals", type=str,
        choices=['none', 'embed', 'gpt', 'oracle'], help="which type of manuals to pass to the model")
    parser.add_argument("--gpt_groundings_path", default="chatgpt_groundings/chatgpt_grounding_for_text_all.json", type=str, help="path to chatgpt groundings")

    # World model arguments
    parser.add_argument("--load_model_from", default=None, help="Path to world model state dict.")
    parser.add_argument("--hidden_size", default=512, type=int, help="World model hidden size.")
    parser.add_argument('--attr_embed_dim', type=int, default=256, help='attribute embedding size')
    parser.add_argument('--action_embed_dim', type=int, default=256, help='action embedding size')
    parser.add_argument('--desc_key_dim', type=int, default=256, help="description key size")
    parser.add_argument('--keep_entity_features_for_parsed_manuals', type=int, default=1)

    parser.add_argument("--learning_rate", default=0.0001, type=float, help="World model learning rate.")
    parser.add_argument("--weight_decay", default=0, type=float, help="World model weight decay.")
    parser.add_argument("--reward_loss_weight", default=1, type=float, help="World model reward loss weight.")
    parser.add_argument("--done_loss_weight", default=1, type=float, help="World model done loss weight.")

    # Dataset arguments
    parser.add_argument("--dataset_path", default="custom_dataset/dataset_64x.pickle", help="path to the dataset file")

    # Training arguments
    parser.add_argument("--train_start_state", default="initial", choices=["initial", "anywhere"], help="Which state that rollouts should start from during training")
    parser.add_argument("--max_rollout_length", default=32, type=int, help="Max length of a rollout to train for")
    parser.add_argument("--update_step", default=32, type=int, help="Number of steps before model update")
    parser.add_argument("--batch_size", default=32, type=int, help="batch_size of training input")
    parser.add_argument("--max_time", default=1000, type=float, help="max train time in hrs")
    parser.add_argument("--max_step", default=1e6, type=int, help="max training step")

    # Logging arguments
    parser.add_argument('--eval_step', default=16384, type=int, help='number of steps between evaluations')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='batch_size for evaluation')
    parser.add_argument('--n_frames', default=64, type=int, help='number of frames to visualize')
    parser.add_argument('--entity', type=str, help="entity to log runs to on wandb")
    parser.add_argument('--mode', type=str, default='online', choices=['online', 'offline'], help='mode to run wandb in')
    parser.add_argument('--use_wandb', type=int, default=0, help='log to wandb?')

    args = parser.parse_args()

    # set output name
    if args.output is None:
        args.output = os.path.join('experiments', args.exp_name)
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    assert args.eval_step % args.update_step == 0

    args.device = torch.device(f"cuda:{args.device}")

    # seed everything
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # start wandb logging
    if args.use_wandb:
        wandb.init(
            project = "messenger",
            entity = args.entity,
            name = args.exp_name + '_' + str(int(time.time())),
            mode = args.mode,
        )
        wandb.config.update(args)

    print(pprint.pformat(vars(args), indent=2))

    # train
    train(args)
