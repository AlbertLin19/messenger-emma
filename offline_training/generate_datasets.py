'''
Collect same_world rollouts of Stage 2 using a random policy and store into a pickle with the following format:
{
    'train_all': {
        'manuals': [...],
        'ground_truths': [...],
        'grid_sequences': [[...], ...],
        'action_sequences': [[...], ...],
    },
    'val_same_worlds': ...,            // entity-role pairings are new
    'test_same_worlds-se': ...,           
    'test_same_worlds': ...,           // ALSO can include any combo of mvmt types
}
'''

import gym
import messenger
import pickle
import random
import numpy as np

from tqdm import tqdm

# make the environments
train_all_env = gym.make(f'msgr-train_all-v2', fix_order=True)
val_same_worlds_env = gym.make(f'msgr-val_same_worlds-v2', fix_order=True)
test_same_worlds_se_env = gym.make(f'msgr-test_same_worlds-se-v2', fix_order=True)
test_same_worlds_env = gym.make(f'msgr-test_same_worlds-v2', fix_order=True)

def wrap_obs(obs):
    """ Convert obs format returned by gym env (dict) to a numpy array expected by model
    """
    return np.concatenate(
        (obs["entities"], obs["avatar"]), axis=-1
    )


def collect_rollouts(env):
    manuals = []
    ground_truths = []
    grid_sequences = []
    action_sequences = []
    pbar = tqdm(total=len(env.all_games) * len(env.game_variants) * len(env.init_states))
    while True:
        obs, text = env.reset(no_type_p=0, attach_ground_truth=True)
        manual = [sent[0] for sent in text]
        ground_truth = [sent[1] for sent in text]
        obs = wrap_obs(obs)

        manuals.append(manual)
        ground_truths.append(ground_truth)
        grid_sequence = [obs]
        action_sequence = []
        
        done = False
        while not done:
            action = random.choice(range(5))
            obs, reward, done, _ = env.step(action)
            obs = wrap_obs(obs)
            grid_sequence.append(obs)
            action_sequence.append(action)
        action_sequence.append(0)

        grid_sequences.append(grid_sequence)
        action_sequences.append(action_sequence)
        pbar.update(1)
        if (env.next_game_idx == 0) and (env.next_variant_idx == 0) and (env.next_init_state_idx == 0):
            break
    pbar.close()
    return manuals, ground_truths, grid_sequences, action_sequences

train_all_manuals, train_all_ground_truths, train_all_grid_sequences, train_all_action_sequences = collect_rollouts(train_all_env)
val_same_worlds_manuals, val_same_worlds_ground_truths, val_same_worlds_grid_sequences, val_same_worlds_action_sequences = collect_rollouts(val_same_worlds_env)
test_same_worlds_se_manuals, test_same_worlds_se_ground_truths, test_same_worlds_se_grid_sequences, test_same_worlds_se_action_sequences = collect_rollouts(test_same_worlds_se_env)
test_same_worlds_manuals, test_same_worlds_ground_truths, test_same_worlds_grid_sequences, test_same_worlds_action_sequences = collect_rollouts(test_same_worlds_env)

dataset = {
    'train_all': {
        'manuals': train_all_manuals,
        'ground_truths': train_all_ground_truths,
        'grid_sequences': train_all_grid_sequences,
        'action_sequences': train_all_action_sequences,
    },
    'val_same_worlds': {
        'manuals': val_same_worlds_manuals,
        'ground_truths': val_same_worlds_ground_truths,
        'grid_sequences': val_same_worlds_grid_sequences,
        'action_sequences': val_same_worlds_action_sequences,
    },
    'test_same_worlds-se': {
        'manuals': test_same_worlds_se_manuals,
        'ground_truths': test_same_worlds_se_ground_truths,
        'grid_sequences': test_same_worlds_se_grid_sequences,
        'action_sequences': test_same_worlds_se_action_sequences,
    },
    'test_same_worlds': {
        'manuals': test_same_worlds_manuals,
        'ground_truths': test_same_worlds_ground_truths,
        'grid_sequences': test_same_worlds_grid_sequences,
        'action_sequences': test_same_worlds_action_sequences,
    },
}

with open('./datasets/stage_2_same_worlds_dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)