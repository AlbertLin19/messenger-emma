'''
Script for training models on stage 1, where the agent either starts with or with the goal
and the sole objective is to interact with the correct item.
'''

import argparse
import time
import pickle
import random
import hashlib

import gym
import messenger # this needs to be imported even though its not used to register gym environment ids
from messenger.models.utils import Encoder
from messenger.envs.config import NPCS
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import wandb
import numpy as np

from model import TrainEMMA, Memory
from train_tools import ObservationBuffer, PPO, TrainStats
from world_model.model import WorldModel
from world_model.utils import ground, key_attend, value_attend


def wrap_obs(obs):
    """ Convert obs format returned by gym env (dict) to a numpy array expected by model
    """
    return np.concatenate(
        (obs["entities"], obs["avatar"]), axis=-1
    )


def train(args):
    model_kwargs = {
        "hist_len": args.hist_len,
        "n_latent_var": args.latent_vars,
        "emb_dim": args.emb_dim,
    }

    optim_kwargs = {
        "weight_decay": args.weight_decay
    }

    ppo = PPO(
        ModelCls = TrainEMMA,
        model_kwargs = model_kwargs,
        device = args.device,
        lr = args.lr,
        gamma = args.gamma,
        K_epochs = args.k_epochs,
        eps_clip = args.eps_clip,
        load_state = args.load_state,
        optim_kwargs=optim_kwargs,
        optimizer=args.optimizer
    )

    # freeze attention mechanism
    if args.freeze_attention:
        ppo.policy.freeze_attention()
        ppo.policy_old.freeze_attention()

    # freeze policy entirely
    if args.freeze_policy:
        for param in ppo.policy.parameters():
            param.requires_grad = False 
        for param in ppo.policy_old.parameters():
            param.requires_grad = False 

    # memory stores all the information needed by PPO to compute losses and make updates
    memory = Memory()

    if args.world_model_train_separately:
        emma = TrainEMMA(**model_kwargs).to(args.device) 
    else:
        emma = ppo.policy

    if args.world_model_train:
        world_model = WorldModel(
            emma=emma,
            val_emb_dim=args.world_model_value_emb_dim,
            latent_size=args.world_model_latent_size,
            hidden_size=args.world_model_hidden_size,
            learning_rate=args.world_model_learning_rate,
            loss_type=args.world_model_loss_type,
            pred_multilabel_threshold=args.world_model_pred_multilabel_threshold,
            refine_pred_multilabel=args.world_model_refine_pred_multilabel,
            val_mlp_attention=args.world_model_value_mlp_attention,
            device=args.device
        )

    # logging variables
    teststats = []
    runstats = []
        
    # make the environments
    env = gym.make(f'msgr-train-v{args.stage}')
    eval_env = gym.make(f'msgr-val-v{args.stage}')

    # training stat tracker
    eval_stats = TrainStats({-1: 'val_death', 1: "val_win"})
    train_stats = TrainStats({-1: 'death', 1: "win"})

    # Text Encoder
    encoder_model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(model=encoder_model, tokenizer=tokenizer, device=args.device, max_length=36)

    # Observation Buffer
    buffer = ObservationBuffer(device=args.device, buffer_size=args.hist_len)

    # training variables
    i_episode = 0
    timestep = 0
    updatestep = 0
    max_win = -1
    max_train_win = -1
    start_time = time.time()

    # used to switch phases of training policy and world model, if specified
    policy_phase = True

    colors = torch.tensor([
        [0, 0, 0], # 0
        [255, 0, 0], # 1
        [255, 85, 0], # 2
        [255, 170, 0], # 3
        [255, 255, 0], # 4
        [170, 255, 0], # 5
        [85, 255, 0], # 6
        [0, 255, 0], # 7
        [0, 255, 85], # 8
        [0, 255, 170], # 9
        [0, 255, 255], # 10
        [0, 170, 255], # 11
        [0, 85, 255], # 12
        [0, 0, 255], # 13
        [85, 0, 255], # 14
        [170, 0, 255], # 15
        [255, 0, 255], # 16
    ])

    while True: # main training loop
        obs, text = env.reset(no_type_p=args.no_type_p)
        obs = wrap_obs(obs)
        if args.world_model_train:
            tensor_obs = torch.from_numpy(obs).long().to(args.device)
        text, _ = encoder.encode(text)
        buffer.reset(obs)
        if args.world_model_train:
            world_model.real_state_reset(tensor_obs)
            world_model.imag_state_reset(tensor_obs)

        # Episode loop
        for t in range(args.max_steps):
            timestep += 1

            if args.world_model_train:
                old_tensor_obs = tensor_obs

            # Running policy_old:
            if args.freeze_policy:
                with torch.no_grad():
                    action = ppo.policy_old.act(buffer.get_obs(), text, memory)
            else:
                action = ppo.policy_old.act(buffer.get_obs(), text, memory)
            obs, reward, done, _ = env.step(action)
            obs = wrap_obs(obs)
            if args.world_model_train:
                tensor_obs = torch.from_numpy(obs).long().to(args.device)

            # World model predictions
            if args.world_model_train:
                world_model.real_step(old_tensor_obs, text, action, tensor_obs)
                with torch.no_grad():
                    world_model.imag_step(text, action, tensor_obs)
            
            # add the step penalty
            reward -= abs(args.step_penalty)

            # add rewards to memory and stats
            if t == args.max_steps - 1 and reward != 1:
                reward = -1.0 # failed to complete objective
                done = True
                
            train_stats.step(reward)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update the model and world_model if its time
            if timestep % args.update_timestep == 0:
                updatestep += 1
                if args.world_model_train:
                    if not args.world_model_train_alternatingly or not policy_phase:
                        world_model.real_loss_update()
                    world_model.real_state_detach()
                    world_model.imag_state_detach()
                    real_loss_and_metrics = world_model.real_loss_and_metrics_reset()
                    imag_loss_and_metrics = world_model.imag_loss_and_metrics_reset()
                if not args.freeze_policy and (not args.world_model_train_alternatingly or policy_phase):
                    policy_loss = ppo.update(memory)
                memory.clear_memory()
                policy_phase = not policy_phase
                timestep = 0
                if updatestep % args.log_loss_interval == 0:
                    updatelog = {'step': train_stats.total_steps}
                    if args.world_model_train:
                        updatelog.update(real_loss_and_metrics)
                        updatelog.update(imag_loss_and_metrics)
                    if not args.freeze_policy:
                        updatelog.update({'policy_loss': policy_loss})
                    wandb.log(updatelog)
                    updatestep = 0
                
            if done:
                break
                
            buffer.update(obs)

        train_stats.end_of_episode()
        i_episode += 1

        # check if max_time has elapsed
        if time.time() - start_time > 60 * 60 * args.max_time:
            break

        # logging
        if i_episode % args.log_interval == 0:
            print("Episode {} \t {}".format(i_episode, train_stats))
            runstats.append(train_stats.compress())
            if not args.check_script:
                wandb.log(runstats[-1])

            episodelog = {'step': train_stats.total_steps}
            
            true_real_probs = F.pad(torch.stack(world_model.true_real_probs, dim=0), (0, 0, 1, 1, 1, 1))
            pred_real_probs = F.pad(torch.stack(world_model.pred_real_probs, dim=0), (0, 0, 1, 1, 1, 1))
            real_probs = torch.cat((true_real_probs, pred_real_probs), dim=2)
            episodelog.update({f'real_prob_{i}': wandb.Video((255*real_probs[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
            episodelog.update({'real_probs': wandb.Video(torch.sum(real_probs.unsqueeze(-1)*colors, dim=-2).permute(0, 3, 1, 2).to(torch.uint8))})

            true_real_multilabels = F.pad(torch.stack(world_model.true_real_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
            pred_real_multilabels = F.pad(torch.stack(world_model.pred_real_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
            real_multilabels = torch.cat((true_real_multilabels, pred_real_multilabels), dim=2)
            episodelog.update({f'real_multilabel_{i}': wandb.Video((255*real_multilabels[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
            episodelog.update({'real_multilabels': wandb.Video(torch.min(torch.sum(real_multilabels.unsqueeze(-1)*colors, dim=-2), torch.tensor([255])).permute(0, 3, 1, 2).to(torch.uint8))})

            true_imag_probs = F.pad(torch.stack(world_model.true_imag_probs, dim=0), (0, 0, 1, 1, 1, 1))
            pred_imag_probs = F.pad(torch.stack(world_model.pred_imag_probs, dim=0), (0, 0, 1, 1, 1, 1))
            imag_probs = torch.cat((true_imag_probs, pred_imag_probs), dim=2)
            episodelog.update({f'imag_prob_{i}': wandb.Video((255*imag_probs[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
            episodelog.update({'imag_probs': wandb.Video(torch.sum(imag_probs.unsqueeze(-1)*colors, dim=-2).permute(0, 3, 1, 2).to(torch.uint8))})

            true_imag_multilabels = F.pad(torch.stack(world_model.true_imag_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
            pred_imag_multilabels = F.pad(torch.stack(world_model.pred_imag_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
            imag_multilabels = torch.cat((true_imag_multilabels, pred_imag_multilabels), dim=2)
            episodelog.update({f'imag_multilabel_{i}': wandb.Video((255*imag_multilabels[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
            episodelog.update({'imag_multilabels': wandb.Video(torch.min(torch.sum(imag_multilabels.unsqueeze(-1)*colors, dim=-2), torch.tensor([255])).permute(0, 3, 1, 2).to(torch.uint8))})

            wandb.log(episodelog)

            entity_ids = [entity.id for entity in NPCS]
            entity_descriptors = {entity.id: None for entity in NPCS}
            entity_names = {entity.id: entity.name for entity in NPCS}
            while None in entity_descriptors.values():
                if hasattr(env, 'cur_env'):
                    if random.random() < env.prob_env_1:
                        cur_env = env.env_1 
                    else:
                        cur_env = env.env_2
                else:
                    cur_env = env
                game = random.choice(cur_env.all_games)
                variant = random.choice(cur_env.game_variants)
                if entity_descriptors[game.enemy.id] is None:
                    entity_descriptors[game.enemy.id] = cur_env.text_manual.get_descriptor(entity=game.enemy.name, entity_type=variant.enemy_type, role="enemy", no_type_p=args.no_type_p)
                if entity_descriptors[game.message.id] is None:
                    entity_descriptors[game.message.id] = cur_env.text_manual.get_descriptor(entity=game.message.name, entity_type=variant.message_type, role="message", no_type_p=args.no_type_p)
                if entity_descriptors[game.goal.id] is None:
                    entity_descriptors[game.goal.id] = cur_env.text_manual.get_descriptor(entity=game.goal.name, entity_type=variant.goal_type, role="goal", no_type_p=args.no_type_p)
            
            ordered_entity_ids = []
            ordered_entity_descriptors = []
            ordered_entity_names = []
            for i in range(17):
                if i in entity_ids:
                    ordered_entity_ids.append(i)
                    ordered_entity_descriptors.append(entity_descriptors[i])
                    ordered_entity_names.append(entity_names[i])
            with torch.no_grad():
                ordered_entity_descriptors, ordered_entity_tokens = encoder.encode(ordered_entity_descriptors)

                policy_key_attention = key_attend(ordered_entity_descriptors, ppo.policy).squeeze(-1).cpu()
                policy_value_attention = value_attend(ordered_entity_descriptors, ppo.policy).squeeze(-1).cpu()
                policy_grounding = ground(ordered_entity_descriptors, ppo.policy)[ordered_entity_ids].cpu()

                world_model_key_attention = key_attend(ordered_entity_descriptors, world_model.emma).squeeze(-1).cpu()
                world_model_value_attention = value_attend(ordered_entity_descriptors, world_model).squeeze(-1).cpu()
                world_model_grounding = ground(ordered_entity_descriptors, world_model.emma)[ordered_entity_ids].cpu()

                attention_table = []
                for i in range(len(ordered_entity_tokens)):
                    for j in range(len(ordered_entity_tokens[i])):
                        attention_table.append([train_stats.total_steps, ordered_entity_tokens[i][j], policy_key_attention[i, j].item(), policy_value_attention[i, j].item(), world_model_key_attention[i, j].item(), world_model_value_attention[i, j].item()])

            wandb.log({
                'step': train_stats.total_steps,
                'policy_grounding': wandb.Image(policy_grounding.unsqueeze(0)),
                'world_model_grounding': wandb.Image(world_model_grounding.unsqueeze(0)),
                'token_attention': wandb.Table(columns=["step", "token", "policy_key", "policy_value", "world_model_key", "world_model_value"], data=attention_table)
            })
            
            if train_stats.compress()['win'] > max_train_win:
                torch.save(ppo.policy_old.state_dict(), args.output + "_maxtrain.pth")
                torch.save(world_model.state_dict(), args.output + "_worldmodel_maxtrain.pth")
                max_train_win = train_stats.compress()['win']
                
            train_stats.reset()
        world_model.vis_logs_reset()

        # run evaluation
        if i_episode % args.eval_interval == 0:
            eval_stats.reset()
            ppo.policy_old.eval()
            if args.world_model_train:
                world_model.eval()

            for eval_episode in range(args.eval_eps):
                obs, text = eval_env.reset(no_type_p=args.no_type_p)
                obs = wrap_obs(obs)
                if args.world_model_train:
                    tensor_obs = torch.from_numpy(obs).long().to(args.device)
                text, _ = encoder.encode(text)
                buffer.reset(obs)
                if args.world_model_train:
                    world_model.real_state_reset(tensor_obs)
                    world_model.imag_state_reset(tensor_obs)

                # Running policy_old:
                for t in range(args.max_steps):
                    if args.world_model_train:
                        old_tensor_obs = tensor_obs
                    with torch.no_grad():
                        action = ppo.policy_old.act(buffer.get_obs(), text, None)
                    obs, reward, done, _ = eval_env.step(action)
                    obs = wrap_obs(obs)
                    if args.world_model_train:
                        tensor_obs = torch.from_numpy(obs).long().to(args.device)

                    # World model predictions
                    if args.world_model_train and (eval_episode >= (args.eval_eps - args.eval_world_model_metrics_eps)):
                        with torch.no_grad():
                            world_model.real_step(old_tensor_obs, text, action, tensor_obs)
                            world_model.imag_step(text, action, tensor_obs)

                    if t == args.max_steps - 1 and reward != 1:
                        reward = -1.0 # failed to complete objective
                        done = True
                        
                    eval_stats.step(reward)
                    if done:
                        break
                    buffer.update(obs)

                if args.world_model_train and (eval_episode < (args.eval_eps - args.eval_world_model_vis_eps)):
                    world_model.vis_logs_reset()
                eval_stats.end_of_episode()

            ppo.policy_old.train()
            if args.world_model_train:
                world_model.train()

            print("TEST: \t {}".format(eval_stats))
            teststats.append(eval_stats.compress(append={"step": train_stats.total_steps}))
            if not args.check_script:
                wandb.log(teststats[-1])

            if args.world_model_train:
                evallog = {'step': train_stats.total_steps}
                
                real_loss_and_metrics = world_model.real_loss_and_metrics_reset()
                imag_loss_and_metrics = world_model.imag_loss_and_metrics_reset()
                evallog.update({f'val_{key}': value for key, value in real_loss_and_metrics.items()})
                evallog.update({f'val_{key}': value for key, value in imag_loss_and_metrics.items()})

                true_real_probs = F.pad(torch.stack(world_model.true_real_probs, dim=0), (0, 0, 1, 1, 1, 1))
                pred_real_probs = F.pad(torch.stack(world_model.pred_real_probs, dim=0), (0, 0, 1, 1, 1, 1))
                real_probs = torch.cat((true_real_probs, pred_real_probs), dim=2)
                evallog.update({f'val_real_prob_{i}': wandb.Video((255*real_probs[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
                evallog.update({'val_real_probs': wandb.Video(torch.sum(real_probs.unsqueeze(-1)*colors, dim=-2).permute(0, 3, 1, 2).to(torch.uint8))})

                true_real_multilabels = F.pad(torch.stack(world_model.true_real_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
                pred_real_multilabels = F.pad(torch.stack(world_model.pred_real_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
                real_multilabels = torch.cat((true_real_multilabels, pred_real_multilabels), dim=2)
                evallog.update({f'val_real_multilabel_{i}': wandb.Video((255*real_multilabels[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
                evallog.update({'val_real_multilabels': wandb.Video(torch.min(torch.sum(real_multilabels.unsqueeze(-1)*colors, dim=-2), torch.tensor([255])).permute(0, 3, 1, 2).to(torch.uint8))})

                true_imag_probs = F.pad(torch.stack(world_model.true_imag_probs, dim=0), (0, 0, 1, 1, 1, 1))
                pred_imag_probs = F.pad(torch.stack(world_model.pred_imag_probs, dim=0), (0, 0, 1, 1, 1, 1))
                imag_probs = torch.cat((true_imag_probs, pred_imag_probs), dim=2)
                evallog.update({f'val_imag_prob_{i}': wandb.Video((255*imag_probs[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
                evallog.update({'val_imag_probs': wandb.Video(torch.sum(imag_probs.unsqueeze(-1)*colors, dim=-2).permute(0, 3, 1, 2).to(torch.uint8))})

                true_imag_multilabels = F.pad(torch.stack(world_model.true_imag_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
                pred_imag_multilabels = F.pad(torch.stack(world_model.pred_imag_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
                imag_multilabels = torch.cat((true_imag_multilabels, pred_imag_multilabels), dim=2)
                evallog.update({f'val_imag_multilabel_{i}': wandb.Video((255*imag_multilabels[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
                evallog.update({'val_imag_multilabels': wandb.Video(torch.min(torch.sum(imag_multilabels.unsqueeze(-1)*colors, dim=-2), torch.tensor([255])).permute(0, 3, 1, 2).to(torch.uint8))})

                wandb.log(evallog)

                entity_ids = [entity.id for entity in NPCS]
                entity_descriptors = {entity.id: None for entity in NPCS}
                entity_names = {entity.id: entity.name for entity in NPCS}
                while None in entity_descriptors.values():
                    if hasattr(eval_env, 'cur_env'):
                        if random.random() < eval_env.prob_env_1:
                            cur_env = eval_env.env_1 
                        else:
                            cur_env = eval_env.env_2
                    else:
                        cur_env = eval_env
                    game = random.choice(cur_env.all_games)
                    variant = random.choice(cur_env.game_variants)
                    if entity_descriptors[game.enemy.id] is None:
                        entity_descriptors[game.enemy.id] = cur_env.text_manual.get_descriptor(entity=game.enemy.name, entity_type=variant.enemy_type, role="enemy", no_type_p=args.no_type_p)
                    if entity_descriptors[game.message.id] is None:
                        entity_descriptors[game.message.id] = cur_env.text_manual.get_descriptor(entity=game.message.name, entity_type=variant.message_type, role="message", no_type_p=args.no_type_p)
                    if entity_descriptors[game.goal.id] is None:
                        entity_descriptors[game.goal.id] = cur_env.text_manual.get_descriptor(entity=game.goal.name, entity_type=variant.goal_type, role="goal", no_type_p=args.no_type_p)
                
                ordered_entity_ids = []
                ordered_entity_descriptors = []
                ordered_entity_names = []
                for i in range(17):
                    if i in entity_ids:
                        ordered_entity_ids.append(i)
                        ordered_entity_descriptors.append(entity_descriptors[i])
                        ordered_entity_names.append(entity_names[i])
                with torch.no_grad():
                    ordered_entity_descriptors, ordered_entity_tokens = encoder.encode(ordered_entity_descriptors)

                    policy_key_attention = key_attend(ordered_entity_descriptors, ppo.policy).squeeze(-1).cpu()
                    policy_value_attention = value_attend(ordered_entity_descriptors, ppo.policy).squeeze(-1).cpu()
                    policy_grounding = ground(ordered_entity_descriptors, ppo.policy)[ordered_entity_ids].cpu()

                    world_model_key_attention = key_attend(ordered_entity_descriptors, world_model.emma).squeeze(-1).cpu()
                    world_model_value_attention = value_attend(ordered_entity_descriptors, world_model).squeeze(-1).cpu()
                    world_model_grounding = ground(ordered_entity_descriptors, world_model.emma)[ordered_entity_ids].cpu()

                    attention_table = []
                    for i in range(len(ordered_entity_tokens)):
                        for j in range(len(ordered_entity_tokens[i])):
                            attention_table.append([train_stats.total_steps, ordered_entity_tokens[i][j], policy_key_attention[i, j].item(), policy_value_attention[i, j].item(), world_model_key_attention[i, j].item(), world_model_value_attention[i, j].item()])

                wandb.log({
                    'step': train_stats.total_steps,
                    'val_policy_grounding': wandb.Image(policy_grounding.unsqueeze(0)),
                    'val_world_model_grounding': wandb.Image(world_model_grounding.unsqueeze(0)),
                    'val_token_attention': wandb.Table(columns=["step", "token", "policy_key", "policy_value", "world_model_key", "world_model_value"], data=attention_table)
                })
                
            if eval_stats.compress()['val_win'] > max_win:
                torch.save(ppo.policy_old.state_dict(), args.output + "_max.pth")
                torch.save(world_model.state_dict(), args.output + "_worldmodel_max.pth")
                max_win = eval_stats.compress()['val_win']
                
            # Save metrics
            with open(args.output + "_metrics.pkl", "wb") as file:
                pickle.dump({"test": teststats, "run": runstats}, file)

            # Save model states
            torch.save(ppo.policy_old.state_dict(), args.output + "_state.pth")
            torch.save(world_model.state_dict(), args.output + "_worldmodel_state.pth")
        world_model.vis_logs_reset()
            
        if i_episode > args.max_eps:
            break
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output", required=True, type=str, help="Local output file name or path.")
    parser.add_argument("--seed", default=None, type=int, help="Set the seed for the model and training.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")

    # Model arguments
    parser.add_argument("--load_state", default=None, help="Path to model state dict.")
    parser.add_argument("--latent_vars", default=128, type=int, help="Latent model dimension.")
    parser.add_argument("--hist_len", default=3, type=int, help="Length of history used by state buffer")
    parser.add_argument("--emb_dim", default=256, type=int, help="embedding size for text")

    # World model arguments
    parser.add_argument("--world_model_train", default=False, action="store_true", help="Whether to train a world model too.")
    parser.add_argument("--world_model_train_alternatingly", default=False, action="store_true", help="Whether to have separate training phases for policy and world model.")
    parser.add_argument("--world_model_train_separately", default=False, action="store_true", help="Whether to have separate networks for policy and world model.")
    parser.add_argument("--world_model_pred_multilabel_threshold", default=0.2, type=float, help="Probability threshold to predict existence of a sprite in pred_multilabel.")
    parser.add_argument("--world_model_refine_pred_multilabel", default=False, action="store_true", help="Whether to refine pred_multilabel by keeping <=1 of each sprite, exactly 1 avatar, and no entities known to not exist in the level.")
    parser.add_argument("--world_model_load_state", default=None, help="Path to world model state dict.")
    parser.add_argument("--world_model_value_emb_dim", default=256, type=int, help="World model value embedding dimension.")
    parser.add_argument("--world_model_value_mlp_attention", default=False, action="store_true", help="Whether to use an MLP to attend to the value tokens.")
    parser.add_argument("--world_model_latent_size", default=512, type=int, help="World model latent size.")
    parser.add_argument("--world_model_hidden_size", default=1024, type=int, help="World model hidden size.")
    parser.add_argument("--world_model_learning_rate", default=0.0005, type=float, help="World model learning rate.")
    parser.add_argument("--world_model_loss_type", default="cross_entropy", choices=["binary_cross_entropy", "cross_entropy"], help="Which loss to use.")
    
    # Environment arguments
    parser.add_argument("--stage", default=1, type=int, help="the stage to run experiment on")
    parser.add_argument("--max_steps", default=4, type=int, help="Maximum num of steps per episode")
    parser.add_argument("--step_penalty", default=0.0, type=float, help="negative reward for each step")
    parser.add_argument("--no_type_p", default=0.15, type=float, help="the probability of getting no movement type info in manual description")
    
    # Training arguments
    parser.add_argument("--update_timestep", default=64, type=int, help="Number of steps before model update")
    parser.add_argument("--lr", default=0.00005, type=float, help="learning rate")
    parser.add_argument("--gamma", default=0.8, type=float, help="discount factor")
    parser.add_argument("--k_epochs", default=4, type=int, help="num epochs to update")
    parser.add_argument("--eps_clip", default=0.1, type=float, help="clip param for PPO")
    parser.add_argument("--optimizer", default="Adam", type=str, help="optimizer class to use")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay for optimizer")
    parser.add_argument("--max_time", default=1000, type=float, help="max train time in hrs")
    parser.add_argument("--max_eps", default=1e10, type=float, help="max training episodes")
    parser.add_argument("--freeze_attention", default=False, action="store_true", help="Do not update attention weights.")
    parser.add_argument("--freeze_policy", default=False, action="store_true", help="Do not update policy network.")

    # Logging arguments
    parser.add_argument('--log_loss_interval', default=50, type=int, help='number of loss updates between logging')
    parser.add_argument('--log_interval', default=500, type=int, help='number of episodes between logging')
    parser.add_argument('--eval_interval', default=500, type=int, help='number of episodes between eval')
    parser.add_argument('--eval_eps', default=500, type=int, help='number of episodes to run eval')
    parser.add_argument('--eval_world_model_vis_eps', default=5, type=int, help='number of episodes to run world model vis eval')
    parser.add_argument('--eval_world_model_metrics_eps', default=50, type=int, help='number of episodes to run world model metrics eval')
    parser.add_argument('--log_group', type=str, help="wandb log group")
    parser.add_argument('--entity', type=str, help="entity to log runs to on wandb")
    parser.add_argument('--check_script', action='store_true', help="run quickly just to see script runs okay.")

    args = parser.parse_args()

    assert args.eval_world_model_metrics_eps >= args.eval_world_model_vis_eps
    
    # get hash of arguments minus seed
    args_dict = vars(args).copy()
    args_dict["device"] = None
    args_dict["seed"] = None
    args_dict["output"] = None
    args_hash = hashlib.md5(
        str(sorted(args_dict.items())).encode("utf-8")
    ).hexdigest()

    args.device = torch.device(f"cuda:{args.device}")

    if args.seed is not None: # seed everything
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    if args.check_script: # for quick debugging
        args.eval_interval = 50
        args.eval_eps = 50
        args.log_interval = 50
        args.max_eps = 100

    else:
        wandb.init(
            project = "msgr-emma",
            entity = args.entity,
            group = args.log_group if args.log_group is not None else args_hash,
            name = f"emma_stage-{args.stage}_seed-{args.seed}"
        )
        wandb.config.update(args)
    
    train(args)