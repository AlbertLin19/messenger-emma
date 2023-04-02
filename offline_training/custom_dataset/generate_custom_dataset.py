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
    'rollouts': {
        split_name: {
            'manual_idxs': [
                [2, 3, 2],
            ],
            'ground_truths': [
                [('mage', 'chasing', 'message'), ('thief', 'fleeing', 'goal'), ('dog', 'chasing', 'enemy')],
            ],
            'grid_sequences': [
                [grid, grid, grid, ...],
            ],
            'action_sequences': [
                [action, action, action, ...],
            ],
            'reward_sequences': [
                [reward, reward, reward, ...],
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

dataset = {
    "texts": texts,
    "rollouts": {},
}

env = gym.make(f'msgr-custom-v2')

for split, games in splits.items():
    print(split)
    manual_idxs = []
    ground_truths = []
    grid_sequences = []
    action_sequences = []
    reward_sequences = []

    for i in tqdm(range(len(games))):
        for _ in range(NUM_REPEATS):
            obs, manual, ground_truth = env.reset(split=split, entities=games[i])
            obs = wrap_obs(obs)
            manual_idx = [texts[ground_truth[j][0]][ground_truth[j][1]][ground_truth[j][2]][split].index(manual[j]) for j in range(len(manual))]
            
            manual_idxs.append(manual_idx)
            ground_truths.append(ground_truth)
            grid_sequence = [obs]
            action_sequence = []
            reward_sequence = []

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
            action_sequence.append(0)
            reward_sequence.append(0)

            grid_sequences.append(grid_sequence)
            action_sequences.append(action_sequence)
            reward_sequences.append(reward_sequence)
    dataset["rollouts"][split] = {
        "manual_idxs": manual_idxs,
        "ground_truths": ground_truths,
        "grid_sequences": grid_sequences,
        "action_sequences": action_sequences,
        "reward_sequences": reward_sequences,
    }

with open(SAVE_PATH, 'wb') as f:
    pickle.dump(dataset, f)