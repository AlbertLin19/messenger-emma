'''
Collect rollouts of custom games of Stage 2 using a random policy and store into a pickle with the following format:
{
    'texts': {
        entity: {
            dynamic: {
                role: {
                    split_name: ["a", "b", "c", ...],
                },
            },
        },
    },
    'keys': {
        'entities': list(texts.keys()),
        'dynamics': list(list(texts.values())[0].keys()),
        'roles': list(list(list(texts.values())[0].values())[0].keys()),
    },
    'rollouts': {
        split_name: {
            'manual_idxs': [
                [2, 3, 2],
            ],
            'ground_truth_idxs': [
                [(2, 1, 1), (11, 1, 2), (10, 2, 0)],
            ],
            'grid_sequences': [
                [grid0, grid1, grid2, ...],
            ],
            'action_sequences': [
                [0, action0-1, action1-2, ...],
            ],
            'reward_sequences': [
                [0, reward1, reward2, ...],
            ],
            'done_sequences': [
                [False, False, False, ...],
            ],
        },
    }
}
'''

import sys

sys.path.append('../..')

import gym
import math
import messenger
import json
import pickle
import random
from collections import defaultdict
from pprint import pprint

import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm

from messenger.envs.config import NPCS, NO_MESSAGE, WITH_MESSAGE

ENTITY_IDS = {entity.name: entity.id for entity in NPCS}
CUSTOM_TO_MESSENGER_ENTITY = {
    "robot": "robot",
    "airplane": "airplane",
    "thief": "thief",
    "scientist": "scientist",
    "queen": "queen",
    "ship": "ship",
    "dog": "dog",
    "bird": "bird",
    "fish": "fish",
    "mage": "mage",
    "orb": "ball",
    "sword": "sword",
}

def wrap_obs(obs, entity_order):
    """ Convert obs format returned by gym env (dict) to a numpy array expected by model
    """

    obs['entities'] = obs['entities'][..., entity_order]

    return np.concatenate((obs["entities"], obs["avatar"]), axis=-1)

random.seed(23)
np.random.seed(23)

SAVE_PATH = "./dataset_shuffle_10k_train_500_eval.pickle"
SPLITS_PATH = "./splits.json"
TEXTS_PATH = "../../messenger/envs/texts/custom_text_splits/custom_text_splits.json"

NUM_TRAIN = 10000
NUM_EVAL = 500


MAX_ROLLOUT_LENGTH = 32

with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

with open(TEXTS_PATH, "r") as f:
    texts = json.load(f)
keys = {
    'entities': list(texts.keys()),
    'dynamics': list(list(texts.values())[0].keys()),
    'roles': list(list(list(texts.values())[0].values())[0].keys()),
}

print(keys)

dataset = {
    "texts": texts,
    "keys": keys,
    "rollouts": {},
}

env = gym.make(f'msgr-custom-v2', shuffle_obs=False)

role_order = {
    'enemy': 0,
    'message': 1,
    'goal': 2
}

for split, games in splits.items():
    combos = set()
    for g in games:
        g = sorted(g, key=lambda e: role_order[e[2]])
        movement_combo = tuple(e[1] for e in g)
        combos.add(movement_combo)
    print(split, len(combos))
    pprint(sorted(combos))


pprint(ENTITY_IDS)

for split, games in splits.items():
    print(split)
    manual_idxs = []
    ground_truth_idxs = []
    grid_sequences = []
    action_sequences = []
    reward_sequences = []
    done_sequences = []

    count_rewards = defaultdict(int)

    if 'train' in split:
        NUM_REPEATS = math.ceil(NUM_TRAIN / len(games))
    else:
        NUM_REPEATS = math.ceil(NUM_EVAL / len(games))

    for i in tqdm(range(len(games))):
        for _ in range(NUM_REPEATS):

            obs, manual, ground_truth = env.reset(split=split, entities=games[i])

            """
            print(ENTITY_IDS)
            print(obs['entities'].reshape(100, 3).max(0))
            print(ground_truth)
            print(manual)
            print()
            """

            # permute observation channels in a consistent order across an episode
            entity_order = np.random.permutation(3)

            # permute order of manual and ground_truth
            manual = [manual[j] for j in entity_order]
            ground_truth = [ground_truth[j] for j in entity_order]

            obs = wrap_obs(obs, entity_order)

            """
            print(ENTITY_IDS)
            print(entity_order)
            print(obs.reshape(100, 4).max(0))
            print(ground_truth)
            print(manual)
            print()
            """

            manual_idx = [texts[ground_truth[j][0]][ground_truth[j][1]][ground_truth[j][2]][split].index(manual[j]) for j in range(len(manual))]
            ground_truth_idx = [(keys['entities'].index(ground_truth[j][0]), keys['dynamics'].index(ground_truth[j][1]), keys['roles'].index(ground_truth[j][2])) for j in range(len(ground_truth))]

            manual_idxs.append(manual_idx)
            ground_truth_idxs.append(ground_truth_idx)

            #print(manual_idx, ground_truth_idx)

            grid_sequence = [obs]
            action_sequence = [0]
            reward_sequence = [0]
            done_sequence = [False]

            done = False
            step = 1
            while not done:
                if step >= MAX_ROLLOUT_LENGTH:
                    break
                step += 1
                action = random.choice(range(5))
                obs, reward, done, _ = env.step(action)
                obs = wrap_obs(obs, entity_order)
                grid_sequence.append(obs)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)

                #print(obs[..., 0].sum(), obs[..., 1].sum(), obs[..., 2].sum())

                #print(step, action, reward, done)
                #print(obs.sum(-1))
                #print()
                #input()
                if reward != 0:
                    count_rewards[reward] += 1

            grid_sequences.append(grid_sequence)
            action_sequences.append(action_sequence)
            reward_sequences.append(reward_sequence)
            done_sequences.append(done_sequence)
    print(split, count_rewards)
    dataset["rollouts"][split] = {
        "manual_idxs": manual_idxs,
        "ground_truth_idxs": ground_truth_idxs,
        "grid_sequences": grid_sequences,
        "action_sequences": action_sequences,
        "reward_sequences": reward_sequences,
        "done_sequences": done_sequences,
    }

#print('DEBUG!!! uncomment saving')
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(dataset, f)
    print('SAVED')
