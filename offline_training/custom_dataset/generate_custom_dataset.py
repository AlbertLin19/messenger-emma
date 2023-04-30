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

INTENTIONS = ['random', 'survive', 'get_message', 'go_to_goal']

actions = [(0, -1, 0), (1, 1, 0), (2, 0, -1), (3, 0, 1), (4, 0, 0)]

def get_avatar_id(obs):
    return obs[..., -1].max()

def get_entity_id_by_role(parsed_manuals, role):
    for e in parsed_manuals:
        if e[2] == role:
            return ENTITY_IDS[CUSTOM_TO_MESSENGER_ENTITY[e[0]]]
    return None

def get_position_by_id(obs, id):
    entity_ids = obs.reshape(100, -1).max(0).tolist()
    c = entity_ids.index(id)
    pos = obs.reshape(100, -1)[:, c].tolist().index(id)
    row = pos // 10
    col = pos % 10
    return row, col

def out_of_bounds(x):
    return x[0] < 0 or x[0] >= 10 or x[1] < 0 or x[1] >= 10

def get_distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def get_best_action_for_chasing(a_pos, t_pos):
    best_d = 1e9
    best_a = None
    # shuffle action order to randomize choice
    for a, dr, dc in random.sample(actions, len(actions)):
        na_pos = (a_pos[0] + dr, a_pos[1] + dc)
        if out_of_bounds(na_pos):
            continue
        d = get_distance(na_pos, t_pos)
        if d < best_d:
            best_d = d
            best_a = a
    return best_a

def get_best_action_for_surviving(a_pos, e_pos, g_pos):
    distance_to_enemy = get_distance(a_pos, e_pos)
    if g_pos is not None:
        distance_to_goal = get_distance(a_pos, g_pos)
    else:
        distance_to_goal = 1e9
    # if far enough from enemy and goal just act randomly
    SAFE_DISTANCE = 6
    if distance_to_enemy >= SAFE_DISTANCE and distance_to_goal >= SAFE_DISTANCE:
        return random.choice(range(len(actions)))
    # otherwise, stay further from both
    best_d = -1e9
    best_a = None
    # shuffle action order to randomize choice
    for a, dr, dc in random.sample(actions, len(actions)):
        na_pos = (a_pos[0] + dr, a_pos[1] + dc)
        if out_of_bounds(na_pos):
            continue
        d = get_distance(na_pos, e_pos)
        if g_pos is not None:
            d = min(d, get_distance(na_pos, g_pos))
        if d >= SAFE_DISTANCE / 2 or d > best_d:
            best_d = d
            best_a = a

    return best_a

def choose_action(obs, parsed_manuals, intention):

    if intention == 'random':
        return random.choice(range(5))

    if intention == 'survive':
        avatar_id = get_avatar_id(obs)
        a_pos = get_position_by_id(obs, avatar_id)
        enemy_id = get_entity_id_by_role(parsed_manuals, 'enemy')
        e_pos = get_position_by_id(obs, enemy_id)
        goal_id = get_entity_id_by_role(parsed_manuals, 'goal')
        g_pos = get_position_by_id(obs, goal_id)
        # if messaged has been obtained, don't care about hitting goal
        if avatar_id == WITH_MESSAGE.id:
            g_pos = None
        # choose action that takes avatar furthest from the enemy
        return get_best_action_for_surviving(a_pos, e_pos, g_pos)

    if intention == 'get_message':
        avatar_id = get_avatar_id(obs)
        # if message has been obtained, act randomly
        if avatar_id == WITH_MESSAGE.id:
            return choose_action(obs, parsed_manuals, 'random')
        a_pos = get_position_by_id(obs, avatar_id)
        message_id = get_entity_id_by_role(parsed_manuals, 'message')
        t_pos = get_position_by_id(obs, message_id)
        # choose action that takes avatar closest to the goal
        return get_best_action_for_chasing(a_pos, t_pos)

    if intention == 'go_to_goal':
        avatar_id = get_avatar_id(obs)
        a_pos = get_position_by_id(obs, avatar_id)
        # if message has been obtained, go to goal
        if avatar_id == WITH_MESSAGE.id:
            goal_id = get_entity_id_by_role(parsed_manuals, 'goal')
            t_pos = get_position_by_id(obs, goal_id)
        # else go to message
        else:
            message_id = get_entity_id_by_role(parsed_manuals, 'message')
            t_pos = get_position_by_id(obs, message_id)
        # choose action that takes avatar closest to the goal or message
        return get_best_action_for_chasing(a_pos, t_pos)

    return None

def wrap_obs(obs, entity_order):
    """ Convert obs format returned by gym env (dict) to a numpy array expected by model
    """
    obs['entities'] = obs['entities'][..., entity_order]
    return np.concatenate((obs["entities"], obs["avatar"]), axis=-1)

random.seed(23)
np.random.seed(23)

SAVE_PATH = "./dataset_shuffle_balanced_intentions_10k_train_500_eval.pickle"
SPLITS_PATH = "./data_splits_final.json"
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

            # permute observation channels in a consistent order across an episode
            entity_order = np.random.permutation(3)

            # permute order of manual and ground_truth
            manual = [manual[j] for j in entity_order]
            ground_truth = [ground_truth[j] for j in entity_order]

            obs = wrap_obs(obs, entity_order)

            manual_idx = [texts[ground_truth[j][0]][ground_truth[j][1]][ground_truth[j][2]][split].index(manual[j]) for j in range(len(manual))]
            ground_truth_idx = [(keys['entities'].index(ground_truth[j][0]), keys['dynamics'].index(ground_truth[j][1]), keys['roles'].index(ground_truth[j][2])) for j in range(len(ground_truth))]

            manual_idxs.append(manual_idx)
            ground_truth_idxs.append(ground_truth_idx)

            grid_sequence = [obs]
            action_sequence = [0]
            reward_sequence = [0]
            done_sequence = [False]

            # choose an intention for the episode
            episode_intention = random.choice(INTENTIONS)
            done = False
            step = 1
            while not done:
                if step >= MAX_ROLLOUT_LENGTH:
                    break
                step += 1
                action = choose_action(obs, ground_truth, episode_intention)
                obs, reward, done, _ = env.step(action)
                obs = wrap_obs(obs, entity_order)
                grid_sequence.append(obs)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)

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
