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
from offline_training.batched_world_model.model_transformer import WorldModel, WorldModelEnv
from dataset import Dataset
from evaluator import Evaluator


def train(args):

    world_model = WorldModel(args).to(args.device)
    print(world_model)

    if args.load_model_from:
        model_state_dict = torch.load(args.load_model_from)
        world_model.load_state_dict(model_state_dict)
        print('Loaded model from', args.load_model_from)

    # Text Encoder
    manual_encoder_model = AutoModel.from_pretrained('bert-base-uncased')
    manual_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    manual_encoder = BatchedEncoder(
        model=manual_encoder_model,
        tokenizer=manual_tokenizer,
        device=args.device,
        max_length=36
    )

    dataset = Dataset(args, seed=args.seed)

    args.eval_every = len(dataset['train_games']) // (args.gradient_accum_iters * args.batch_size)
    print(pprint.pformat(vars(args), indent=2))

    train_iter = dataset['train_games'].iterate_batches(batch_size=args.batch_size, cycle=True)
    train_stats = defaultdict(list)

    best_metric = defaultdict(lambda: defaultdict(lambda: 1e9))

    for i, batch in zip(range(int(args.max_step)), train_iter):

        # EVALUATION
        if i % args.eval_every == 0:
            wandb_stats = {}
            wandb_stats['step'] = i
            log_str = []
            avg_train_metric = {}
            for k in train_stats:
                avg_train_metric[k] = np.average(train_stats[k])
                log_str.append('%s %.4f' % (k, avg_train_metric[k]))
                wandb_stats['train/' + k] = avg_train_metric[k]
            log_str = ', '.join(log_str)
            print()
            print('After %d step' % i)
            print('  TRAIN', log_str)
            print()

            # reset train metrics
            train_stats = defaultdict(list)

            for eval_split in dataset:

                if 'dev' not in eval_split:
                    continue

                eval_iter = dataset[eval_split].iterate_batches(batch_size=args.eval_batch_size, cycle=False)

                with torch.no_grad():
                    eval_stats = evaluate(
                        eval_iter,
                        args,
                        eval_split,
                        world_model,
                        manual_encoder,
                        best_metric[eval_split]
                    )
                '''
                for i in range(4):
                    print(eval_stats['entity_%d_id_loss_len_0' % i])
                '''

                for k in eval_stats:
                    wandb_stats[('%s/' % eval_split) + k] = eval_stats[k]
                    wandb_stats[('%s_best/' % eval_split) + k] = best_metric[eval_split][k]

            if args.use_wandb:
                wandb.log(wandb_stats)

        if args.eval_mode:
            break

        # TRAIN
        world_model.reset(is_eval=False)

        input_manual = get_manual(args, batch, manual_encoder)
        true_parsed_manual = batch['true_parsed_manual']

        T = batch['grid'].shape[0]
        #T = 2
        for t in range(T - 1):

            mask = batch['mask'][t]
            grid = batch['grid'][t]
            reward = batch['reward'][t]
            done = batch['done'][t]
            action = batch['action'][t + 1]
            next_state_description = batch['state_description'][t + 1]

            world_model(
                t,
                input_manual,
                true_parsed_manual,
                grid,
                reward,
                done,
                action,
                next_state_description,
                mask
            )

        train_loss = world_model.learn()
        for k in train_loss:
            train_stats[k].append(train_loss[k])

        if (i + 1) % args.gradient_accum_iters == 0:
            world_model.update_params()



def evaluate(data_iter, args, split, world_model, manual_encoder, best_metric):

    eval_stats = defaultdict(list)
    for i, batch in enumerate(data_iter):

        debug = None

        print(batch['id'])
        try:
            debug = batch['id'].index('train_games_9856')
        except:
            pass
        print(debug)

        world_model.reset(is_eval=True)

        input_manual = get_manual(args, batch, manual_encoder)
        true_parsed_manual = batch['true_parsed_manual']

        T = batch['grid'].shape[0]
        #T = 2
        for t in range(T - 1):
            mask = batch['mask'][t]
            grid = batch['grid'][t]
            reward = batch['reward'][t]
            done = batch['done'][t]
            action = batch['action'][t + 1]
            next_state_description = batch['state_description'][t + 1]

            if debug is not None:
                print(grid[debug].sum(-1))
                print(grid[debug].view(-1, 4).max(0)[0])
                print(next_state_description[debug][1:].tolist())
                print(args.description_tokenizer.decode(next_state_description[debug][1:].tolist()))

            logit, target = world_model(
                t,
                input_manual,
                true_parsed_manual,
                grid,
                reward,
                done,
                action,
                next_state_description,
                mask,
                debug=debug
            )

            add_eval_stats(t, eval_stats, logit, target)

    logged_losses = ['total_loss', 'state_loss', 'reward_loss', 'done_loss']

    avg_metric = {}
    avg_metric['total_loss'] = 0
    for k in eval_stats:
        avg_metric[k] = np.average(eval_stats[k])
        avg_metric[k.replace('loss', 'perp')] = math.exp(avg_metric[k])
        if k in logged_losses:
            avg_metric['total_loss'] += avg_metric[k]

    # update best model
    for k in avg_metric:
        if avg_metric[k] < best_metric[k]:
            best_metric[k] = avg_metric[k]
            if k in logged_losses:
                model_path = os.path.join(args.output, '%s_best_%s.ckpt' % (split, k))
                if not args.eval_mode:
                    torch.save(world_model.state_dict(), model_path)
                    print('Saved best %s %s to %s' % (split, k, model_path))

    print('  EVALUATION on %s' % split)
    log_str = []
    for k in logged_losses:
        if k in avg_metric:
            log_str.append('%s %.4f' % (k, avg_metric[k]))
    log_str = '    CURRENT ' + ', '.join(log_str)
    print(log_str)
    log_str = []
    for k in logged_losses:
        if k in best_metric:
            log_str.append('%s %.4f' % (k, best_metric[k]))
    log_str = '    BEST    ' + ', '.join(log_str)
    print(log_str)

    return avg_metric

def get_manual(args, batch, manual_encoder):
    if args.manuals == 'oracle':
        return batch['true_parsed_manual']
    if args.manuals == 'gpt':
        return batch['gpt_parsed_manual']
    if args.manuals == 'embed':
        embedded_manual, _ = manual_encoder.encode(batch['manual'])
        return embedded_manual
    return None

def add_eval_stats(timestep, stats, logit, target):

    for i in range(4):

        start = i * 4
        end = (i + 1) * 4

        logit['entity_%d' % i] = logit['state'][:, start:end]
        target['entity_%d' % i] = target['state'][:, start:end]

        logit['entity_%d_id' % i] = logit['state'][:, start + 1:start + 2]
        target['entity_%d_id' % i] = target['state'][:, start + 1:start + 2]

        logit['entity_%d_loc' % i] = logit['state'][:, start + 2:end]
        target['entity_%d_loc' % i] = target['state'][:, start + 2:end]

    for k in logit:
        stats[k + '_loss'].extend(get_loss_list(logit[k], target[k]))

    if timestep <= 10:
        stats['state_loss_len_%d' % timestep].extend(get_loss_list(logit['state'], target['state']))
        for t in range(1, timestep + 1):
            stats['state_loss_len_upto_%d' % t].extend(get_loss_list(logit['state'], target['state']))

        for i in range(4):
            stats['entity_%d_loc_loss_len_%d' % (i, timestep)].extend(
                get_loss_list(logit['entity_%d_loc' % i], target['entity_%d_loc' % i]))
            stats['entity_%d_id_loss_len_%d' % (i, timestep)].extend(
                get_loss_list(logit['entity_%d_id' % i], target['entity_%d_id' % i]))



def get_loss_list(logit, target):
    logit = logit.flatten(0, logit.dim() - 2)
    target = target.flatten()
    loss_list = F.cross_entropy(logit, target, reduction='none', ignore_index=-1)
    non_pad_ids = (target != -1).nonzero().squeeze(-1)
    loss_list = loss_list[non_pad_ids].tolist()
    return loss_list

def step_through(args):

    world_model = WorldModel(args).to(args.device)
    print(world_model)

    if args.load_model_from:
        model_state_dict = torch.load(args.load_model_from)
        world_model.load_state_dict(model_state_dict)
        print('Loaded model from', args.load_model_from)

    # Text Encoder
    manual_encoder_model = AutoModel.from_pretrained('bert-base-uncased')
    manual_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    manual_encoder = BatchedEncoder(
        model=manual_encoder_model,
        tokenizer=manual_tokenizer,
        device=args.device,
        max_length=36
    )

    env = WorldModelEnv(world_model)
    dataset = Dataset(args, seed=args.seed)

    split = 'train_dev_games'
    while True:
        batch = dataset[split].random_batch(args.batch_size)
        print(batch['id'][0])
        grid = batch['grid'][0]
        manual = batch['gpt_parsed_manual']

        ob = env.reset(grid, manual)

        print(ob[0].sum(-1), manual[0])

        for t in range(args.max_rollout_length):
            action = batch['action'][t + 1]
            ob, reward, done = env.step(action)
            if done: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--output', default=None, type=str, help='Local output file name or path.')
    parser.add_argument('--seed', default=123, type=int, help='Set the seed for the model and training.')
    parser.add_argument('--device', default=0, type=int, help='cuda device ordinal to train on.')
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--eval_mode', type=int, default=0, help='evaluation mode')
    parser.add_argument('--step_through_mode', type=int, default=0, help='step through each action')
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
    parser.add_argument('--max_step', default=1e6, type=int, help='max training step')

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

    args.description_tokenizer = Tokenizer.from_file(args.description_tokenizer_file)

    print(pprint.pformat(vars(args), indent=2))

    if args.step_through_mode:
        step_through(args)
    else:
        train(args)
