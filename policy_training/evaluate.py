'''
Script for evaluating policy and world_model_informed_policy on stage 2.
'''

import os
import argparse
from argparse import Namespace
import json
import time
import pickle
import random

import gym
import messenger # this needs to be imported even though its not used to register gym environment ids
from messenger.models.utils import Encoder
from messenger.models.emma import EMMA
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

from offline_training.batched_world_model.model_khanh import WorldModel
from messenger.models.utils import ObservationBuffer

import matplotlib.pyplot as plt
from tqdm import tqdm

from offline_training.chatgpt_groundings.utils import ENTITY_GROUNDING_LOOKUP, MOVEMENT_GROUNDING_LOOKUP, ROLE_GROUNDING_LOOKUP

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

def evaluate(args):

    # load policy
    policy = EMMA(
        hist_len=args.hist_len,
        n_latent_var=args.latent_vars,
        emb_dim=args.emb_dim,
    ).to(args.device)
    policy.load_state_dict(torch.load(args.load_state, map_location=args.device))
    policy.eval()

    # load world model
    world_model_args = {
        key[len("world_model_"):]: value for key, value in vars(args).items() if key[:len("world_model_")] == "world_model_"
    }
    world_model_args.update({
        "batch_size": args.num_policy_samples,
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

    # load splits
    with open(args.splits_path, 'r') as f:
        split_games = json.load(f)
    splits = list(split_games.keys())
    
    # make the environment
    env = gym.make(f'msgr-custom-v2', shuffle_obs=False)

    # Text Encoder
    encoder_model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(model=encoder_model, tokenizer=tokenizer, device=args.device, max_length=36)

    # chatgpt groundings
    with open(args.world_model_gpt_groundings_path, "r") as f:
        gpt_groundings = json.load(f)
        # convert groundings into keywords
        for e, grounding in gpt_groundings.items():
            gpt_groundings[e] = [ENTITY_GROUNDING_LOOKUP[grounding[0]], MOVEMENT_GROUNDING_LOOKUP[grounding[1]], ROLE_GROUNDING_LOOKUP[grounding[2]]]


    # Observation Buffer
    buffer = ObservationBuffer(device=args.device, buffer_size=args.hist_len)

    for split in splits:
        print('evaluating', split)
        if 'train' in split:
            print('skipping...')
            continue
        policy_total_rewards = []
        policy_with_world_model_total_rewards = []
        for episode in tqdm(range(len(split_games[split]))):
            
            # evaluate policy
            obs, manual, _ = env.reset(split=split, entities=split_games[split][episode])
            buffer.reset(obs)

            # episode loop
            total_reward = 0
            for t in range(args.max_steps):

                with torch.no_grad():
                    action = policy(buffer.get_obs(), manual, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                    
                if done:
                    break
                    
                buffer.update(obs)
            policy_total_rewards.append(total_reward)
            print(" policy alone:", total_reward)

            # evaluate policy with world model
            obs, manual, _ = env.reset(split=split, entities=split_games[split][episode])
            buffer.reset(obs)
            grid = torch.from_numpy(wrap_obs(obs)).long().to(args.device)
            grids = grid.expand(args.num_policy_samples, *grid.shape)
            world_model.state_reset(grids)

            parsed_manual = [gpt_groundings[e] for e in manual]
            parsed_manuals = [parsed_manual for _ in range(args.num_policy_samples)]

            # episode loop
            total_reward = 0
            for t in range(args.max_steps):
                
                with torch.no_grad():
                    hidden_states, cell_states = torch.clone(world_model.hidden_states), torch.clone(world_model.cell_states)
                    
                    # FIND BEST POLICY ACTION TO TAKE ACCORDING TO WORLD MODEL
                    imagined_initial_actions = None
                    imagined_total_rewards = torch.zeros(args.num_policy_samples).to(args.device)
                    imagined_dones = torch.zeros(args.num_policy_samples).to(args.device)

                    imagined_buffers = [ObservationBuffer(device=args.device, buffer_size=args.hist_len) for _ in range(args.num_policy_samples)]
                    for imagined_buffer in imagined_buffers:
                        imagined_buffer.buffer = [buffer_obs for buffer_obs in buffer.buffer]
                    imagined_grids = grids

                    for _ in range(args.max_lookahead_length):
                        imagined_actions = torch.tensor([policy(imagined_buffer.get_obs(), manual, temperature=args.policy_temperature) for imagined_buffer in imagined_buffers]).long().to(args.device)
                        if imagined_initial_actions is None:
                            imagined_initial_actions = imagined_actions
                        imagined_preds = world_model.pred(
                            imagined_grids,
                            parsed_manuals,
                            imagined_actions,
                            sample=True
                        )
                        for i in range(args.num_policy_samples):
                            imagined_buffers[i].update(unwrap_grid(imagined_preds["grid"][i]))
                        imagined_grids = imagined_preds["grid"]

                        imagined_total_rewards += torch.logical_not(imagined_dones)*imagined_preds["reward"]
                        imagined_dones = torch.logical_or(imagined_dones, imagined_preds["done"])
                        if imagined_dones.all():
                            break

                    # REVERT HIDDEN STATES AND CELL STATES
                    world_model.hidden_states, world_model.cell_states = hidden_states, cell_states

                action = imagined_initial_actions[torch.argmax(imagined_total_rewards).item()]
                actions = torch.tensor([action for _ in range(args.num_policy_samples)]).long().to(args.device)
                with torch.no_grad():
                    world_model.pred(
                        grids,
                        parsed_manuals,
                        actions,
                        sample=False
                    )
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                    
                if done:
                    break
                    
                buffer.update(obs)
                grid = torch.from_numpy(wrap_obs(obs)).long().to(args.device)
                grids = grid.expand(args.num_policy_samples, *grid.shape)
                
            policy_with_world_model_total_rewards.append(total_reward)
            print(" policy with world model:", total_reward)
        
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

    # Environment arguments
    parser.add_argument("--splits_path", default="../offline_training/custom_dataset/data_splits_final_with_test.json", help="path to data splits")
    
    # Evaluation arguments
    parser.add_argument("--policy_temperature", default=2, type=float, help="temperature of the policy (logits scaling)")
    parser.add_argument("--num_policy_samples", default=64, type=int, help="number of policy samples to evaluate")
    parser.add_argument("--max_lookahead_length", default=32, type=int, help="maximum steps to lookahead")
    parser.add_argument("--max_steps", default=64, type=int, help="max length of an episode")

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device}")
    if args.output_folder is None:
        args.output_folder = f"evaluation/{os.path.basename(args.load_state).split('.')[0]}_{os.path.basename(args.world_model_load_model_from).split('.')[0]}_{time.time()}/"
    os.makedirs(args.output_folder)
        

    # seed everything
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    evaluate(args)