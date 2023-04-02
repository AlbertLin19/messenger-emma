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
        },
    }
}
'''

import gym
import messenger
import json
import pickle
import random
import numpy as np

from tqdm import tqdm

def wrap_obs(obs):
    """ Convert obs format returned by gym env (dict) to a numpy array expected by model
    """
    return np.concatenate(
        (obs["entities"], obs["avatar"]), axis=-1
    )

SAVE_PATH = "./dataset.pickle"
SPLITS_PATH = "./splits.json"
TEXTS_PATH = "../../messenger/envs/texts/custom_text_splits/custom_text_splits.json"

NUM_REPEATS = 32
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

dataset = {
    "texts": texts,
    "keys": keys,
    "rollouts": {},
}

env = gym.make(f'msgr-custom-v2')

for split, games in splits.items():
    print(split)
    manual_idxs = []
    ground_truth_idxs = []
    grid_sequences = []
    action_sequences = []
    reward_sequences = []

    for i in tqdm(range(len(games))):
        for _ in range(NUM_REPEATS):
            obs, manual, ground_truth = env.reset(split=split, entities=games[i])
            obs = wrap_obs(obs)
            manual_idx = [texts[ground_truth[j][0]][ground_truth[j][1]][ground_truth[j][2]][split].index(manual[j]) for j in range(len(manual))]
            ground_truth_idx = [(keys['entities'].index(ground_truth[j][0]), keys['dynamics'].index(ground_truth[j][1]), keys['roles'].index(ground_truth[j][2])) for j in range(len(ground_truth))]
            
            manual_idxs.append(manual_idx)
            ground_truth_idxs.append(ground_truth_idx)
            grid_sequence = [obs]
            action_sequence = [0]
            reward_sequence = [0]

            done = False
            step = 1
            while not done:
                if step >= MAX_ROLLOUT_LENGTH:
                    break 
                step += 1
                action = random.choice(range(5))
                obs, reward, done, _ = env.step(action)
                obs = wrap_obs(obs)
                grid_sequence.append(obs)
                action_sequence.append(action)
                reward_sequence.append(reward)

            grid_sequences.append(grid_sequence)
            action_sequences.append(action_sequence)
            reward_sequences.append(reward_sequence)
    dataset["rollouts"][split] = {
        "manual_idxs": manual_idxs,
        "ground_truth_idxs": ground_truth_idxs,
        "grid_sequences": grid_sequences,
        "action_sequences": action_sequences,
        "reward_sequences": reward_sequences,
    }

with open(SAVE_PATH, 'wb') as f:
    pickle.dump(dataset, f)