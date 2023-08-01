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
from tokenizers import Tokenizer

sys.path.append('..')

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from messenger.models.utils import BatchedEncoder
from offline_training.batched_world_model.tokenizer import Tokenizer
from dataset import Dataset
from evaluator import Evaluator


def train(args):

    model = Tokenizer(args).to(args.device)
    print(model)

    if args.load_model_from:
        model_state_dict = torch.load(args.load_model_from)
        model.load_state_dict(model_state_dict)
        print('Loaded model from', args.load_model_from)

    dataset = Dataset(args, seed=args.seed)

    args.eval_every = len(dataset['train_games']) // args.batch_size
    print(pprint.pformat(vars(args), indent=2))

    train_iter = dataset['train_games'].iterate_batches(batch_size=args.batch_size, cycle=True)
    train_stats = defaultdict(list)

    best_stats = defaultdict(lambda: defaultdict(lambda: 1e9))

    for i, batch in zip(range(int(args.max_step)), train_iter):

        if i % args.eval_every == 0:
            train_avg_stats, train_log_str = average_and_make_log_str(train_stats)
            print()
            print('After %d step' % i)
            print('  TRAIN', train_log_str)
            print()

            wandb_stats = {}
            wandb_stats['step'] = i
            for k in train_avg_stats:
                wandb_stats['train/' + k] = train_avg_stats[k]

            train_stats = defaultdict(list)

            for split in dataset:

                if 'dev' not in split:
                    continue

                eval_iter = dataset[split].iterate_batches(batch_size=args.eval_batch_size, cycle=False)
                eval_stats = defaultdict(list)
                for j, eval_batch in enumerate(eval_iter):

                    eval_input_grids = make_input(eval_batch)
                    with torch.no_grad():
                        eval_loss = model.learn(eval_input_grids, is_eval=True)

                        if args.inspect_mode and j % 10 == 0:
                            out, logit = model(eval_input_grids)
                            recon = logit.argmax(-1)
                            idx = random.randint(0, recon.shape[0] - 1)
                            print(split)
                            print(eval_input_grids[idx].sum(-1))
                            print(out.tokens[idx].tolist())
                            print(recon[idx].sum(-1))

                    for k in eval_loss:
                        eval_stats[k].append(eval_loss[k])

                eval_avg_stats, eval_log_str = average_and_make_log_str(eval_stats)
                print('%30s' % split, eval_log_str)

                for k in eval_avg_stats:
                    if eval_avg_stats[k] < best_stats[split][k]:
                        best_stats[split][k] = eval_avg_stats[k]
                        save_path = os.path.join(args.output, '%s_best_%s.ckpt' % (split, k))
                        torch.save(model.state_dict(), save_path)
                        print('Saved best %s model to %s' % (split, save_path))

                for k in eval_avg_stats:
                    wandb_stats['%s/%s' % (split, k)] = eval_avg_stats[k]
                    wandb_stats['%s_best/%s' % (split, k)] = best_stats[split][k]

            for split in dataset:

                if 'test' in split:
                    continue

                best_log_str = []
                for k in best_stats[split]:
                    best_log_str.append('%s %.4f' % (k, best_stats[split][k]))
                best_log_str = ', '.join(best_log_str)
                print('%30s' % ('best ' + split), best_log_str)

            if args.use_wandb:
                wandb.log(wandb_stats)

        if args.eval_mode:
            break

        # TRAIN
        input_grids = make_input(batch)

        train_loss = model.learn(input_grids)

        for k in train_loss:
            train_stats[k].append(train_loss[k])


def make_input(batch):

    input_grids = []
    for i in range(args.batch_size):
        input_grids.append(batch['grid'][i, :batch['len'][i]])
    input_grids = torch.cat(input_grids, dim=0)

    return input_grids


def average_and_make_log_str(stats):
    avg_stats = {}
    log_str = []
    for k in stats:
        avg_stats[k] = np.average(stats[k])
        log_str.append('%s %.4f' % (k, avg_stats[k]))
    log_str = ', '.join(log_str)
    return avg_stats, log_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--output', default=None, type=str, help='Local output file name or path.')
    parser.add_argument('--seed', default=123, type=int, help='Set the seed for the model and training.')
    parser.add_argument('--device', default=0, type=int, help='cuda device ordinal to train on.')
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--eval_mode', type=int, default=0, help='evaluation mode')
    parser.add_argument('--inspect_mode', type=int, default=0)
    parser.add_argument('--description_tokenizer_file', type=str, help='Tokenizer file')
    parser.add_argument('--transformer_config_file', type=str, help='Transformer configuration file')

    parser.add_argument('--debug_latent_loss_only', type=int, default=0)
    parser.add_argument('--debug_no_latent_loss', type=int, default=0)
    parser.add_argument('--debug_zero_latent', type=int, default=0)
    parser.add_argument('--debug_no_reward_done_input', type=int, default=1)
    parser.add_argument('--debug_no_latent', type=int, default=0)
    parser.add_argument('--debug_no_predict_other_ids', type=int, default=0)
    parser.add_argument('--debug_no_manual_features', type=int, default=0)

    # text config
    parser.add_argument('--manuals', type=str,
        choices=['none', 'embed', 'gpt', 'oracle'], help='which type of manuals to pass to the model')
    parser.add_argument('--gpt_groundings_path', default='chatgpt_groundings/chatgpt_grounding_for_text_all.json', type=str, help='path to chatgpt groundings')

    # World model arguments
    parser.add_argument('--load_model_from', default=None, help='Path to world model state dict.')
    parser.add_argument('--hidden_dim', default=512, type=int, help='World model hidden size.')
    parser.add_argument('--attr_embed_dim', type=int, default=256, help='attribute embedding size')
    parser.add_argument('--action_embed_dim', type=int, default=256, help='action embedding size')
    parser.add_argument('--desc_key_dim', type=int, default=256, help='description key size')
    parser.add_argument('--keep_entity_features_for_parsed_manuals', type=int, default=1)


    parser.add_argument('--id_embed_dim', default=64, type=int)
    parser.add_argument('--codebook_size', default=128, type=int)
    parser.add_argument('--codebook_embed_dim', default=64, type=int)
    parser.add_argument('--z_channels', default=64, type=int)
    parser.add_argument('--with_lpips', default=0, type=int)

    parser.add_argument('--latent_dim', type=int, default=64)

    parser.add_argument('--transformer_n_layer', type=int, default=-1)
    parser.add_argument('--transformer_n_embd', type=int, default=-1)
    parser.add_argument('--transformer_n_head', type=int, default=-1)

    parser.add_argument('--learning_rate', default=0.0001, type=float, help='World model learning rate.')
    parser.add_argument('--weight_decay', default=0, type=float, help='World model weight decay.')
    parser.add_argument('--reward_loss_weight', default=1, type=float, help='World model reward loss weight.')
    parser.add_argument('--done_loss_weight', default=1, type=float, help='World model done loss weight.')
    parser.add_argument('--gradient_accum_iters', default=2, type=int, help='Number of gradient accumulation iterations')

    # Dataset arguments
    parser.add_argument('--dataset_path', default='custom_dataset/dataset_64x.pickle', help='path to the dataset file')

    # Training arguments
    parser.add_argument('--train_start_state', default='initial', choices=['initial', 'anywhere'], help='Which state that rollouts should start from during training')
    parser.add_argument('--max_rollout_length', default=32, type=int, help='Max length of a rollout to train for')
    parser.add_argument('--update_step', default=32, type=int, help='Number of steps before model update')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size of training input')
    parser.add_argument('--max_time', default=1000, type=float, help='max train time in hrs')
    parser.add_argument('--max_step', default=80000, type=int, help='max training step')

    # Logging arguments
    parser.add_argument('--eval_every', default=300, type=int, help='number of steps between evaluations')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='batch_size for evaluation')
    parser.add_argument('--n_frames', default=64, type=int, help='number of frames to visualize')
    parser.add_argument('--entity', type=str, help='entity to log runs to on wandb')
    parser.add_argument('--mode', type=str, default='online', choices=['online', 'offline'], help='mode to run wandb in')
    parser.add_argument('--use_wandb', type=int, default=0, help='log to wandb?')

    args = parser.parse_args()

    # set output name
    if args.output is None:
        args.output = os.path.join('experiments', args.exp_name)
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    args.device = torch.device('cuda')

    # seed everything
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # start wandb logging
    if args.use_wandb:
        wandb.init(
            project = 'messenger',
            entity = args.entity,
            name = args.exp_name + '_' + str(int(time.time())),
            mode = args.mode,
        )
        wandb.config.update(args)

    train(args)
