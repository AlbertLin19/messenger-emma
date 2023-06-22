'''
Script for evaluating policy and world_model_informed_policy on stage 2.
'''

import os
import sys

sys.path.append('..')

import argparse
from argparse import Namespace
import json
import time
import pickle
import random
from copy import deepcopy as dc
import pprint

import gym
import messenger # this needs to be imported even though its not used to register gym environment ids
from messenger.models.utils import Encoder
from messenger.models.emma import EMMA
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

from offline_training.batched_world_model.model_new import WorldModel
from messenger.models.utils import ObservationBuffer

import matplotlib.pyplot as plt
from tqdm import tqdm

from offline_training.chatgpt_groundings.utils import ENTITY_GROUNDING_LOOKUP, MOVEMENT_GROUNDING_LOOKUP, ROLE_GROUNDING_LOOKUP

class Policy:

    def __init__(self, policy):
        self.policy = policy

    def reset(self, obs, manual):
        pass

    def __call__(self, obs, buffer, manual, deterministic=False):
        obs_hist = buffer.get_obs()
        return self.policy(obs_hist, manual, deterministic=deterministic)

class PolicyWithWorldModel:

    def __init__(self, args, explore_policy):
        self.explore_policy = explore_policy
        self.device = args.device
        self.num_actions = 5
        self.buffer_hist_len = args.hist_len
        self.num_simulations = args.num_policy_samples
        self.max_lookahead_length = args.max_lookahead_length
        self.temp = args.policy_temperature
        self.batch_size = args.world_model_batch_size
        self.discount_factor = 0.9

        assert self.batch_size % self.num_actions == 0
        assert self.num_simulations % self.batch_size == 0

        self.world_model = self._load_world_model(args)
        self.gpt_groundings = self._load_gpt_groundings(args)

    def _load_world_model(self, args):
        # load world model
        world_model_args = {
            key[len("world_model_"):]: value for key, value in vars(args).items() \
                if key[:len("world_model_")] == "world_model_"
        }
        world_model_args.update({
            "device": args.device,
            "learning_rate": 0,
            "weight_decay": 0,
            "reward_loss_weight": 0,
            "done_loss_weight": 0,
            "loss_weights": {
                'loc': 0,
                'id': 0,
                'reward': 0,
                'done': 0
            },
        })
        world_model = WorldModel(Namespace(**world_model_args)).to(args.device)
        world_model.load_state_dict(torch.load(args.world_model_load_model_from, map_location=args.device))
        world_model.eval()
        return world_model

    def _load_gpt_groundings(self, args):
        with open(args.world_model_gpt_groundings_path, "r") as f:
            gpt_groundings = json.load(f)
            # convert groundings into keywords
            for e, grounding in gpt_groundings.items():
                gpt_groundings[e] = [
                    ENTITY_GROUNDING_LOOKUP[grounding[0]],
                    MOVEMENT_GROUNDING_LOOKUP[grounding[1]],
                    ROLE_GROUNDING_LOOKUP[grounding[2]]
                ]
        return gpt_groundings

    def _make_grids(self, obs):
        grid = torch.from_numpy(wrap_obs(obs)).long().to(self.device)
        grids = grid.expand(self.batch_size, *grid.shape)
        return grids

    def _reset_world_model(self, obs):
        grids = self._make_grids(obs)
        self.world_model.state_reset(grids)

    def _simulate_and_evaluate(self, manual, init_grids, init_buffer):
        rewards = [[] for _ in range(self.num_actions)]
        # save world model state
        h, c = self.world_model.hidden_states.clone(), self.world_model.cell_states.clone()
        for k in range(self.num_simulations // self.batch_size):
            # create buffers, rewards, and dones for this simulation
            sim_buffers = []
            for i in range(self.batch_size):
                buffer = ObservationBuffer(self.buffer_hist_len, self.device)
                buffer.buffer = dc(init_buffer.buffer)
                sim_buffers.append(buffer)
            sim_rewards = [[] for _ in range(self.batch_size)]
            sim_dones = [[] for _ in range(self.batch_size)]
            sim_has_dones = [False] * self.batch_size

            # first, try taking EVERY action
            init_actions = torch.arange(self.num_actions).to(self.device)
            init_actions = init_actions.unsqueeze(0).repeat(self.batch_size // self.num_actions, 1)
            init_actions = init_actions.view(-1)
            with torch.no_grad():
                preds = self.world_model.pred(
                    init_grids,
                    self.parsed_manuals,
                    init_actions,
                    sample=True
                )
            # update buffers, rewards, and dones
            grids = preds['grid']
            """
            print('==============')
            print(init_grids[0].sum(-1))
            print(ENTITY_GROUNDING_LOOKUP)
            print(self.parsed_manuals[0])
            print(preds['done'][0].item())
            print(preds['reward'][0].item())
            print(grids[0].sum(-1))
            input()
            """
            for i in range(self.batch_size):
                sim_buffers[i].update(unwrap_grid(grids[i]))
                sim_dones[i].append(preds['done'][i].item())
                sim_rewards[i].append(preds['reward'][i].item())
                sim_has_dones[i] |= sim_dones[i][-1]
            # rollout with policy
            for _ in range(self.max_lookahead_length):
                actions = []
                for buffer in sim_buffers:
                    obs_hist = buffer.get_obs()
                    with torch.no_grad():
                        a = self.explore_policy(
                            obs_hist,
                            manual,
                            deterministic=False,
                            temperature=self.temp
                        )
                    actions.append(a)
                actions = torch.tensor(actions).to(self.device)
                with torch.no_grad():
                    preds = self.world_model.pred(
                        grids,
                        self.parsed_manuals,
                        actions,
                        sample=True
                    )
                # update buffers, rewards, and dones
                grids = preds['grid']
                """
                print(ENTITY_GROUNDING_LOOKUP)
                print(self.parsed_manuals[0])
                print(actions[0])
                print(preds['done'][0].item())
                print(preds['reward'][0].item())
                print(grids[0].sum(-1))
                input()
                """
                for i in range(self.batch_size):
                    sim_buffers[i].update(unwrap_grid(grids[i]))
                    sim_dones[i].append(preds['done'][i].item())
                    sim_rewards[i].append(preds['reward'][i].item())
                    sim_has_dones[i] |= sim_dones[i][-1]

                if all(sim_has_dones):
                    break

            for i in range(self.batch_size):
                assert len(sim_dones[i]) == len(sim_rewards[i])
                l = len(sim_dones[i])
                for j in range(len(sim_dones[i])):
                    if sim_dones[i][j]:
                        l = j + 1
                        break
                total_reward = 0
                for j in reversed(range(l)):
                    total_reward = total_reward * self.discount_factor + sim_rewards[i][j]
                rewards[i % self.num_actions].append(total_reward)

            # revert world model back to original state
            self.world_model.hidden_states, self.world_model.cell_states = h, c

        for i in range(self.num_actions):
            rewards[i] = np.average(rewards[i])

        return rewards

    def reset(self, obs, manual):
        # parse manual
        parsed_manual = [self.gpt_groundings[e] for e in manual]
        self.parsed_manuals = [parsed_manual for _ in range(self.batch_size)]
        # reset world model
        self._reset_world_model(obs)

    def __call__(self, obs, buffer, manual, deterministic=False):
        grids = self._make_grids(obs)
        rewards = self._simulate_and_evaluate(manual, grids, buffer)
        best_action = np.argmax(np.array(rewards))
        # print(grids[0].sum(-1))
        # print(best_action)
        best_actions = best_action * torch.ones(self.batch_size).to(self.device).long()
        with torch.no_grad():
            preds = self.world_model.pred(
                grids,
                self.parsed_manuals,
                best_actions,
                sample=False # not important
            )
        # print(preds['reward'][0])
        return best_action


def wrap_obs(obs):
    """ Convert obs format returned by gym env (dict) to a numpy array expected by model
    """
    return np.concatenate((obs["entities"], obs["avatar"]), axis=-1)

def unwrap_grid(grid):
    """ Convert grid format returned by world model to an obs expected by policy
    """
    return {
        "entities": grid[..., :-1].detach().cpu().numpy(),
        "avatar": grid[..., -1:].detach().cpu().numpy()
    }

def rollout(env, policy, buffer, split, game):
    # evaluate policy
    obs, manual, _ = env.reset(split=split, entities=game)
    buffer.reset(obs)
    policy.reset(obs, manual)

    # episode loop
    total_reward = 0
    for t in range(args.max_steps):
        with torch.no_grad():
            action = policy(obs, buffer, manual, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done: break
        buffer.update(obs)

    print('#########################')

    return total_reward


def evaluate(args):

    # load policy
    emma_policy = EMMA(
        hist_len=args.hist_len,
        n_latent_var=args.latent_vars,
        emb_dim=args.emb_dim,
    ).to(args.device)
    emma_policy.load_state_dict(torch.load(args.load_state, map_location=args.device))
    emma_policy.eval()

    policy = Policy(emma_policy)
    policy_world_model = PolicyWithWorldModel(args, emma_policy)

    # load splits
    with open(args.splits_path, 'r') as f:
        split_games = json.load(f)
    splits = list(split_games.keys())

    # make the environment
    env = gym.make(f'msgr-custom-v2', shuffle_obs=False)

    # Observation Buffer
    buffer = ObservationBuffer(device=args.device, buffer_size=args.hist_len)

    for split in splits:
        print('evaluating', split)
        if 'train' in split or 'dev' in split:
            print('skipping...')
            continue

        policy_total_rewards = []
        policy_with_world_model_total_rewards = []
        for i, episode in enumerate(range(len(split_games[split]))):

            if i >= args.max_episodes:
                break

            print(i)
            game = split_games[split][episode]

            total_reward = rollout(env, policy, buffer, split, game)
            policy_total_rewards.append(total_reward)
            print(" policy alone:", total_reward, 'avg: ', np.average(policy_total_rewards))

            # evaluate policy with world model
            total_reward = rollout(env, policy_world_model, buffer, split, game)
            policy_with_world_model_total_rewards.append(total_reward)
            print(" policy with world model:", total_reward, 'avg: ', np.average(policy_with_world_model_total_rewards))

        # save to file
        with open(os.path.join(args.output_folder, f"{split}.json"), "w") as f:
            json.dump(
                {
                    "policy_total_rewards": policy_total_rewards,
                    "policy_with_world_model_total_rewards": policy_with_world_model_total_rewards
                }, f
            )

        policy_total_rewards = np.array(policy_total_rewards)
        policy_mean = policy_total_rewards.mean()
        policy_std = policy_total_rewards.std()

        policy_with_world_model_total_rewards = np.array(policy_with_world_model_total_rewards)
        policy_with_world_model_mean = policy_with_world_model_total_rewards.mean()
        policy_with_world_model_std = policy_with_world_model_total_rewards.std()

        # plot and save
        x = ["Policy Alone", "Policy with World Model"]
        y = [policy_mean, policy_with_world_model_mean]
        yerr = [policy_std, policy_with_world_model_std]
        plt.figure()
        plt.bar(x, y)
        plt.errorbar(x, y, yerr=yerr, fmt="o", color="black")
        plt.title(f"Total Reward on Split: {split}")
        plt.ylabel("Total Reward")
        plt.savefig(os.path.join(args.output_folder, f"{split}.jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output_folder", default=None, type=str, help="output folder")
    parser.add_argument("--seed", default=0, type=int, help="Set the seed for the model and training.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")

    # Policy arguments
    parser.add_argument("--load_state", default="stage_1/output/emma_s1_1_train_games_max.pth", help="Path to model state dict.")
    parser.add_argument("--latent_vars", default=128, type=int, help="Latent model dimension.")
    parser.add_argument("--hist_len", default=3, type=int, help="Length of history used by state buffer")
    parser.add_argument("--emb_dim", default=256, type=int, help="embedding size for text")

    # World model arguments
    parser.add_argument("--world_model_manuals", default="gpt", type=str,
        choices=['none', 'embed', 'gpt', 'oracle'], help="which type of manuals to pass to the model")
    parser.add_argument("--world_model_gpt_groundings_path", default="../offline_training/chatgpt_groundings/chatgpt_grounding_for_text_all.json", type=str, help="path to chatgpt groundings")
    parser.add_argument("--world_model_load_model_from", default="../offline_training/experiments/gpt_shuffle_balanced_intentions_10k_train_500_eval/ne_nr_or_nm_best_total_loss.ckpt", help="Path to world model state dict.")
    parser.add_argument("--world_model_hidden_size", default=512, type=int, help="World model hidden size.")
    parser.add_argument('--world_model_attr_embed_dim', type=int, default=256, help='attribute embedding size')
    parser.add_argument('--world_model_action_embed_dim', type=int, default=256, help='action embedding size')
    parser.add_argument('--world_model_desc_key_dim', type=int, default=256, help="description key size")
    parser.add_argument('--world_model_keep_entity_features_for_parsed_manuals', type=int, default=1)
    parser.add_argument('--world_model_batch_size', type=int, default=20)

    # Environment arguments
    parser.add_argument("--splits_path", default="../offline_training/custom_dataset/data_splits_final_with_test.json", help="path to data splits")

    # Evaluation arguments
    parser.add_argument("--policy_temperature", default=5, type=float, help="temperature of the policy (logits scaling)")
    parser.add_argument("--num_policy_samples", default=20, type=int, help="number of policy samples to evaluate")
    parser.add_argument("--max_lookahead_length", default=5, type=int, help="maximum steps to lookahead")
    parser.add_argument("--max_steps", default=64, type=int, help="max length of an episode")
    parser.add_argument("--max_episodes", default=100000, type=int)

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device}")
    if args.output_folder is None:
        args.output_folder = f"evaluation/{os.path.basename(args.load_state).split('.')[0]}_policy_{os.path.basename(args.world_model_load_model_from).split('.')[0]}_worldmodel_tp{int(args.policy_temperature)}_ps{args.num_policy_samples}_la{args.max_lookahead_length}/"
    os.makedirs(args.output_folder)

    print(pprint.pformat(vars(args), indent=2))

    # seed everything
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_printoptions(precision=1, sci_mode=False, linewidth=100)

    evaluate(args)
