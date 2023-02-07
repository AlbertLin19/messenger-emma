'''
Script for offline training world models on Stage 2.
'''

import argparse
import time
import pickle
import random
import hashlib
import torch
import torch.nn.functional as F
import wandb
import numpy as np

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from messenger.models.utils import BatchedEncoder
from offline_training.batched_world_model.model import BatchedWorldModel
from dataloader import DataLoader
from analyzer import Analyzer

def train(args):
    world_model = BatchedWorldModel(
        key_type=args.world_model_key_type,
        key_dim=args.world_model_key_dim,
        val_type=args.world_model_val_type,
        val_dim=args.world_model_val_dim,
        latent_size=args.world_model_latent_size,
        hidden_size=args.world_model_hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.world_model_learning_rate,
        prediction_type=args.world_model_prediction_type,
        pred_multilabel_threshold=args.world_model_pred_multilabel_threshold,
        refine_pred_multilabel=args.world_model_refine_pred_multilabel,
        device=args.device
    )
    if args.world_model_key_freeze:
        world_model.freeze_key()
    if args.world_model_val_freeze:
        world_model.freeze_val()

    with open(args.output + '_architecture.txt', 'w') as f:
        f.write(str(world_model))

    # Analyzers
    train_all_real_analyzer = Analyzer("real_", args.eval_length, args.vis_length)
    train_all_imag_analyzer = Analyzer("imag_", args.eval_length, args.vis_length)

    # Text Encoder
    encoder_model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = BatchedEncoder(model=encoder_model, tokenizer=tokenizer, device=args.device, max_length=36)

    # Dataloaders
    print("creating dataloaders")
    train_all_dataloader = DataLoader(args.dataset_path, 'train_all', args.batch_size, args.max_rollout_length)
    val_same_worlds_dataloader = DataLoader(args.dataset_path, 'val_same_worlds', args.batch_size, args.max_rollout_length)
    test_same_worlds_se_dataloader = DataLoader(args.dataset_path, 'test_same_worlds-se', args.batch_size, args.max_rollout_length)
    test_same_worlds_dataloader = DataLoader(args.dataset_path, 'test_same_worlds', args.batch_size, args.max_rollout_length)
    print("finished creating dataloaders")

    # training variables
    step = 0
    start_time = time.time()

    grids, actions, manuals, ground_truths = train_all_dataloader.reset()
    tensor_grids = torch.from_numpy(grids).long().to(args.device)
    tensor_actions = torch.from_numpy(actions).long().to(args.device)
    manuals, _ = encoder.encode(manuals)
    world_model.real_state_reset(tensor_grids, torch.ones(grids.shape[0], dtype=bool))
    world_model.imag_state_reset(tensor_grids, torch.ones(grids.shape[0], dtype=bool))
    pbar = tqdm(total=args.max_step)
    while step < args.max_step: # main training loop
        if args.world_model_key_freeze and ((args.world_model_key_unfreeze_step >= 0) and (step >= args.world_model_key_unfreeze_step)):
            world_model.unfreeze_key()
            args.world_model_key_freeze = False
        if args.world_model_val_freeze and ((args.world_model_val_unfreeze_step >= 0) and (step >= args.world_model_val_unfreeze_step)):
            world_model.unfreeze_val()
            args.world_model_val_freeze = False

        old_tensor_grids = tensor_grids
        grids, actions, manuals, ground_truths, news = train_all_dataloader.step()
        tensor_news = torch.from_numpy(news).bool().to(args.device)
        do_backprops = torch.logical_not(tensor_news)
        tensor_grids = torch.from_numpy(grids).long().to(args.device)
        tensor_actions = torch.from_numpy(actions).long().to(args.device)
        manuals, _ = encoder.encode(manuals)
    
        # accumulate gradient
        if args.world_model_loss_source == "real":
            real_results = world_model.real_step(old_tensor_grids, manuals, ground_truths, tensor_actions, tensor_grids, do_backprops)
            with torch.no_grad():
                imag_results = world_model.imag_step(manuals, ground_truths, tensor_actions, tensor_grids, do_backprops)
        elif args.world_model_loss_source == "imag":
            imag_results = world_model.imag_step(manuals, ground_truths, tensor_actions, tensor_grids, do_backprops)
            with torch.no_grad():
                real_results = world_model.real_step(old_tensor_grids, manuals, ground_truths, tensor_actions, tensor_grids, do_backprops)
        else:
            raise NotImplementedError
        step += 1    

        # perform update
        if step % args.update_step == 0:
            if args.world_model_loss_source == "real":
                world_model.real_loss_update()
            elif args.world_model_loss_source == "imag":
                world_model.imag_loss_update()
            real_loss = world_model.real_loss_reset()
            imag_loss = world_model.imag_loss_reset()
            world_model.real_state_detach()
            world_model.imag_state_detach()
            
            updatelog = {
                "step": step,
                "real_loss": real_loss,
                "imag_loss": imag_loss,
            }
            wandb.log(updatelog)

        # push to analyzers
        if step % args.eval_step > (args.eval_step - args.eval_length):
            train_all_real_analyzer.push(*real_results)
            train_all_imag_analyzer.push(*imag_results)

        # log evaluation
        if step % args.eval_step == 0:
            eval_log = {
                "step": step,
            }
            eval_log.update(train_all_real_analyzer.getLog())
            eval_log.update(train_all_imag_analyzer.getLog())
            wandb.log(eval_log)

        # reset states if new rollouts started
        world_model.real_state_reset(tensor_grids, tensor_news)
        world_model.imag_state_reset(tensor_grids, tensor_news)

        pbar.update(1)

        # check if max_time has elapsed
        if time.time() - start_time > 60 * 60 * args.max_time:
            break
    pbar.close()

    # # logging
    # if i_episode % args.log_interval == 0:
    #     print("Episode {} \t {}".format(i_episode, train_stats))
    #     runstats.append(train_stats.compress())
    #     if not args.check_script:
    #         wandb.log(runstats[-1])

    #     episodelog = {'step': train_stats.total_steps}
        
    #     true_imag_probs = F.pad(torch.stack(world_model.true_imag_probs, dim=0), (0, 0, 1, 1, 1, 1))
    #     pred_imag_probs = F.pad(torch.stack(world_model.pred_imag_probs, dim=0), (0, 0, 1, 1, 1, 1))
    #     imag_probs = torch.cat((true_imag_probs, pred_imag_probs), dim=2)
    #     episodelog.update({f'imag_prob_{i}': wandb.Video((255*imag_probs[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
    #     episodelog.update({'imag_probs': wandb.Video(torch.sum(imag_probs.unsqueeze(-1)*colors, dim=-2).permute(0, 3, 1, 2).to(torch.uint8))})

    #     true_imag_multilabels = F.pad(torch.stack(world_model.true_imag_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
    #     pred_imag_multilabels = F.pad(torch.stack(world_model.pred_imag_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
    #     imag_multilabels = torch.cat((true_imag_multilabels, pred_imag_multilabels), dim=2)
    #     episodelog.update({f'imag_multilabel_{i}': wandb.Video((255*imag_multilabels[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
    #     episodelog.update({'imag_multilabels': wandb.Video(torch.min(torch.sum(imag_multilabels.unsqueeze(-1)*colors, dim=-2), torch.tensor([255])).permute(0, 3, 1, 2).to(torch.uint8))})

    #     wandb.log(episodelog)

    #     entity_names = get_NPCS()
    #     entity_descriptors = {entity_id: None for entity_id in entity_names.keys()}
    #     while None in entity_descriptors.values():
    #         if hasattr(env, 'cur_env'):
    #             if random.random() < env.prob_env_1:
    #                 cur_env = env.env_1 
    #             else:
    #                 cur_env = env.env_2
    #         else:
    #             cur_env = env
    #         game = random.choice(cur_env.all_games)
    #         variant = random.choice(cur_env.game_variants)
    #         if entity_descriptors[game.enemy.id] is None:
    #             entity_descriptors[game.enemy.id] = cur_env.text_manual.get_descriptor(entity=game.enemy.name, entity_type=variant.enemy_type, role="enemy", no_type_p=args.no_type_p, attach_ground_truth=True)
    #         if entity_descriptors[game.message.id] is None:
    #             entity_descriptors[game.message.id] = cur_env.text_manual.get_descriptor(entity=game.message.name, entity_type=variant.message_type, role="message", no_type_p=args.no_type_p, attach_ground_truth=True)
    #         if entity_descriptors[game.goal.id] is None:
    #             entity_descriptors[game.goal.id] = cur_env.text_manual.get_descriptor(entity=game.goal.name, entity_type=variant.goal_type, role="goal", no_type_p=args.no_type_p, attach_ground_truth=True)
        
    #     ordered_entity_ids = []
    #     ordered_entity_descriptors = []
    #     ordered_entity_names = []
    #     for i in range(17):
    #         if i in entity_names.keys():
    #             ordered_entity_ids.append(i)
    #             ordered_entity_descriptors.append(entity_descriptors[i])
    #             ordered_entity_names.append(entity_names[i])
    #     with torch.no_grad():
    #         ordered_ground_truths = [sent[1] for sent in ordered_entity_descriptors]
    #         ordered_entity_descriptors = [sent[0] for sent in ordered_entity_descriptors]
    #         ordered_entity_descriptors, ordered_entity_tokens = encoder.encode(ordered_entity_descriptors)
    #         world_model_grounding = ground(ordered_entity_descriptors, ordered_ground_truths, world_model)[ordered_entity_ids].cpu()

    #         groundinglog = {
    #             'step': train_stats.total_steps,
    #             'world_model_grounding': wandb.Image(world_model_grounding.unsqueeze(0))
    #         }

    #         if ("emma" in args.world_model_key_type) or ("emma" in args.world_model_val_type):
    #             ordered_entity_tokens = np.array(ordered_entity_tokens)
    #             columns = [train_stats.total_steps*np.ones(ordered_entity_tokens.size), ordered_entity_tokens.flatten()]
    #             column_names = ["step", "token"]
    #             if "emma" in args.world_model_key_type:
    #                 world_model_key_attention = world_model.scale_key(ordered_entity_descriptors).squeeze(-1).cpu()
    #                 columns.append(world_model_key_attention.numpy().flatten())
    #                 column_names.append("world_model_key")
    #             if "emma" in args.world_model_val_type:
    #                 world_model_value_attention = world_model.scale_val(ordered_entity_descriptors).squeeze(-1).cpu()
    #                 columns.append(world_model_value_attention.numpy().flatten())
    #                 column_names.append("world_model_value")
    #             attention_table = np.stack(columns, axis=-1)
    #             groundinglog.update({'token_attention': wandb.Table(columns=column_names, data=attention_table)})
                
    #     wandb.log(groundinglog)
        
    #     # if train_stats.compress()['win'] > max_train_win:
    #     #     torch.save(ppo.policy_old.state_dict(), args.output + "_maxtrain.pth")
    #     #     torch.save(world_model.state_dict(), args.output + "_worldmodel_maxtrain.pth")
    #     #     max_train_win = train_stats.compress()['win']
            
    #     train_stats.reset()
    # world_model.vis_logs_reset()

    # # run evaluation
    # if i_episode % args.eval_interval == 0:
    #     # update and clear existing training loss and metrics
    #     if ((args.world_model_loss_source == "real") and (world_model.real_step_count > 0)) or ((args.world_model_loss_source == "imag") and (world_model.imag_step_count > 0)):
    #         if args.world_model_loss_source == "real":
    #             world_model.real_loss_update()
    #         elif args.world_model_loss_source == "imag":
    #             world_model.imag_loss_update()
    #         real_loss_and_metrics = world_model.real_loss_and_metrics_reset()
    #         imag_loss_and_metrics = world_model.imag_loss_and_metrics_reset()
            
    #         updatelog = {'step': train_stats.total_steps}
    #         updatelog.update(real_loss_and_metrics)
    #         updatelog.update(imag_loss_and_metrics)
    #         wandb.log(updatelog)

    #     eval_stats.reset()
    #     if not args.do_nothing_policy:
    #         ppo.policy_old.eval()
    #     world_model.eval()

    #     for eval_episode in range(args.eval_eps):
    #         obs, text = eval_env.reset(no_type_p=args.no_type_p, attach_ground_truth=True)
    #         ground_truth = [sent[1] for sent in text]
    #         text = [sent[0] for sent in text]
    #         obs = wrap_obs(obs)
    #         tensor_obs = torch.from_numpy(obs).long().to(args.device)
    #         text, _ = encoder.encode(text)
    #         buffer.reset(obs)
    #         world_model.real_state_reset(tensor_obs)
    #         world_model.imag_state_reset(tensor_obs)

    #         # Running policy_old:
    #         for t in range(args.max_steps):
    #             old_tensor_obs = tensor_obs
    #             with torch.no_grad():
    #                 if args.do_nothing_policy:
    #                     action = 4
    #                 else:
    #                     action = random.choice(range(5)) if random.random() < args.random_policy_p else ppo.policy_old.act(buffer.get_obs(), text, None)
    #             obs, reward, done, _ = eval_env.step(action)
    #             obs = wrap_obs(obs)
    #             tensor_obs = torch.from_numpy(obs).long().to(args.device)

    #             # World model predictions
    #             if eval_episode >= (args.eval_eps - args.eval_world_model_metrics_eps):
    #                 with torch.no_grad():
    #                     world_model.real_step(old_tensor_obs, text, ground_truth, action, tensor_obs)
    #                     world_model.imag_step(text, ground_truth, action, tensor_obs)

    #             if t == args.max_steps - 1 and reward != 1:
    #                 reward = -1.0 # failed to complete objective
    #                 done = True
                    
    #             eval_stats.step(reward)
    #             if done:
    #                 break
    #             buffer.update(obs)

    #         if eval_episode < (args.eval_eps - args.eval_world_model_vis_eps):
    #             world_model.vis_logs_reset()
    #         eval_stats.end_of_episode()

    #     if not args.do_nothing_policy:
    #         ppo.policy_old.train()
    #     world_model.train()

    #     print("TEST: \t {}".format(eval_stats))
    #     teststats.append(eval_stats.compress(append={"step": train_stats.total_steps}))
    #     if not args.check_script:
    #         wandb.log(teststats[-1])

    #     evallog = {'step': train_stats.total_steps}
            
    #     real_loss_and_metrics = world_model.real_loss_and_metrics_reset()
    #     imag_loss_and_metrics = world_model.imag_loss_and_metrics_reset()
    #     evallog.update({f'val_{key}': value for key, value in real_loss_and_metrics.items()})
    #     evallog.update({f'val_{key}': value for key, value in imag_loss_and_metrics.items()})

    #     true_real_probs = F.pad(torch.stack(world_model.true_real_probs, dim=0), (0, 0, 1, 1, 1, 1))
    #     pred_real_probs = F.pad(torch.stack(world_model.pred_real_probs, dim=0), (0, 0, 1, 1, 1, 1))
    #     real_probs = torch.cat((true_real_probs, pred_real_probs), dim=2)
    #     evallog.update({f'val_real_prob_{i}': wandb.Video((255*real_probs[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
    #     evallog.update({'val_real_probs': wandb.Video(torch.sum(real_probs.unsqueeze(-1)*colors, dim=-2).permute(0, 3, 1, 2).to(torch.uint8))})

    #     true_real_multilabels = F.pad(torch.stack(world_model.true_real_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
    #     pred_real_multilabels = F.pad(torch.stack(world_model.pred_real_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
    #     real_multilabels = torch.cat((true_real_multilabels, pred_real_multilabels), dim=2)
    #     evallog.update({f'val_real_multilabel_{i}': wandb.Video((255*real_multilabels[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
    #     evallog.update({'val_real_multilabels': wandb.Video(torch.min(torch.sum(real_multilabels.unsqueeze(-1)*colors, dim=-2), torch.tensor([255])).permute(0, 3, 1, 2).to(torch.uint8))})

    #     true_imag_probs = F.pad(torch.stack(world_model.true_imag_probs, dim=0), (0, 0, 1, 1, 1, 1))
    #     pred_imag_probs = F.pad(torch.stack(world_model.pred_imag_probs, dim=0), (0, 0, 1, 1, 1, 1))
    #     imag_probs = torch.cat((true_imag_probs, pred_imag_probs), dim=2)
    #     evallog.update({f'val_imag_prob_{i}': wandb.Video((255*imag_probs[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
    #     evallog.update({'val_imag_probs': wandb.Video(torch.sum(imag_probs.unsqueeze(-1)*colors, dim=-2).permute(0, 3, 1, 2).to(torch.uint8))})

    #     true_imag_multilabels = F.pad(torch.stack(world_model.true_imag_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
    #     pred_imag_multilabels = F.pad(torch.stack(world_model.pred_imag_multilabels, dim=0), (0, 0, 1, 1, 1, 1))
    #     imag_multilabels = torch.cat((true_imag_multilabels, pred_imag_multilabels), dim=2)
    #     evallog.update({f'val_imag_multilabel_{i}': wandb.Video((255*imag_multilabels[..., i:i+1]).permute(0, 3, 1, 2).to(torch.uint8)) for i in range(17)})
    #     evallog.update({'val_imag_multilabels': wandb.Video(torch.min(torch.sum(imag_multilabels.unsqueeze(-1)*colors, dim=-2), torch.tensor([255])).permute(0, 3, 1, 2).to(torch.uint8))})

    #     wandb.log(evallog)

    #     entity_names = get_NPCS()
    #     entity_descriptors = {entity_id: None for entity_id in entity_names.keys()}
    #     while None in entity_descriptors.values():
    #         if hasattr(eval_env, 'cur_env'):
    #             if random.random() < eval_env.prob_env_1:
    #                 cur_env = eval_env.env_1 
    #             else:
    #                 cur_env = eval_env.env_2
    #         else:
    #             cur_env = eval_env
    #         game = random.choice(cur_env.all_games)
    #         variant = random.choice(cur_env.game_variants)
    #         if entity_descriptors[game.enemy.id] is None:
    #             entity_descriptors[game.enemy.id] = cur_env.text_manual.get_descriptor(entity=game.enemy.name, entity_type=variant.enemy_type, role="enemy", no_type_p=args.no_type_p, attach_ground_truth=True)
    #         if entity_descriptors[game.message.id] is None:
    #             entity_descriptors[game.message.id] = cur_env.text_manual.get_descriptor(entity=game.message.name, entity_type=variant.message_type, role="message", no_type_p=args.no_type_p, attach_ground_truth=True)
    #         if entity_descriptors[game.goal.id] is None:
    #             entity_descriptors[game.goal.id] = cur_env.text_manual.get_descriptor(entity=game.goal.name, entity_type=variant.goal_type, role="goal", no_type_p=args.no_type_p, attach_ground_truth=True)
        
    #     ordered_entity_ids = []
    #     ordered_entity_descriptors = []
    #     ordered_entity_names = []
    #     for i in range(17):
    #         if i in entity_names.keys():
    #             ordered_entity_ids.append(i)
    #             ordered_entity_descriptors.append(entity_descriptors[i])
    #             ordered_entity_names.append(entity_names[i])
    #     with torch.no_grad():
    #         ground_truths = [sent[1] for sent in ordered_entity_descriptors]
    #         ordered_entity_descriptors = [sent[0] for sent in ordered_entity_descriptors]
    #         ordered_entity_descriptors, ordered_entity_tokens = encoder.encode(ordered_entity_descriptors)
    #         world_model_grounding = ground(ordered_entity_descriptors, ground_truths, world_model)[ordered_entity_ids].cpu()

    #         groundinglog = {
    #             'step': train_stats.total_steps,
    #             'val_world_model_grounding': wandb.Image(world_model_grounding.unsqueeze(0))
    #         }

    #         if ("emma" in args.world_model_key_type) or ("emma" in args.world_model_val_type):
    #             ordered_entity_tokens = np.array(ordered_entity_tokens)
    #             columns = [train_stats.total_steps*np.ones(ordered_entity_tokens.size), ordered_entity_tokens.flatten()]
    #             column_names = ["step", "token"]
    #             if "emma" in args.world_model_key_type:
    #                 world_model_key_attention = world_model.scale_key(ordered_entity_descriptors).squeeze(-1).cpu()
    #                 columns.append(world_model_key_attention.numpy().flatten())
    #                 column_names.append("world_model_key")
    #             if "emma" in args.world_model_val_type:
    #                 world_model_value_attention = world_model.scale_val(ordered_entity_descriptors).squeeze(-1).cpu()
    #                 columns.append(world_model_value_attention.numpy().flatten())
    #                 column_names.append("world_model_value")
    #             attention_table = np.stack(columns, axis=-1)
    #             groundinglog.update({'val_token_attention': wandb.Table(columns=column_names, data=attention_table)})
                
    #     wandb.log(groundinglog)
            
    #     # if eval_stats.compress()['val_win'] > max_win:
    #     #     torch.save(ppo.policy_old.state_dict(), args.output + "_max.pth")
    #     #     torch.save(world_model.state_dict(), args.output + "_worldmodel_max.pth")
    #     #     max_win = eval_stats.compress()['val_win']
            
    #     # # Save metrics
    #     # with open(args.output + "_metrics.pkl", "wb") as file:
    #     #     pickle.dump({"test": teststats, "run": runstats}, file)

    #     # # Save model states
    #     # torch.save(ppo.policy_old.state_dict(), args.output + "_state.pth")
    #     # torch.save(world_model.state_dict(), args.output + "_worldmodel_state.pth")
    # world_model.vis_logs_reset()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output", default=None, type=str, help="Local output file name or path.")
    parser.add_argument("--seed", default=0, type=int, help="Set the seed for the model and training.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")

    # World model arguments
    parser.add_argument("--world_model_pred_multilabel_threshold", default=0.5, type=float, help="Probability threshold to predict existence of a sprite in pred_multilabel when prediction_type == 'existence'.")
    parser.add_argument("--world_model_refine_pred_multilabel", default=True, action="store_true", help="Whether to refine pred_multilabel by keeping <=1 of each sprite, exactly 1 avatar, and no entities known to not exist in the level.")
    parser.add_argument("--world_model_load_state", default=None, help="Path to world model state dict.")
    parser.add_argument("--world_model_key_dim", default=256, type=int, help="World model key embedding dimension.")
    parser.add_argument("--world_model_key_type", default="oracle", choices=["oracle", "emma", "emma-mlp_scale"], help="What to use to process the descriptors' key tokens.")
    parser.add_argument("--world_model_key_freeze", default=False, action="store_true", help="Whether to freeze key module.")
    parser.add_argument("--world_model_key_unfreeze_step", default=5e5, type=int, help="Train step to unfreeze key module, -1 means never.")
    parser.add_argument("--world_model_val_dim", default=256, type=int, help="World model value embedding dimension.")
    parser.add_argument("--world_model_val_type", default="oracle", choices=["oracle", "emma", "emma-mlp_scale"], help="What to use to process the descriptors' value tokens.")
    parser.add_argument("--world_model_val_freeze", default=False, action="store_true", help="Whether to freeze val module.")
    parser.add_argument("--world_model_val_unfreeze_step", default=5e5, type=int, help="Train step to unfreeze val module, -1 means never.")
    parser.add_argument("--world_model_latent_size", default=512, type=int, help="World model latent size.")
    parser.add_argument("--world_model_hidden_size", default=1024, type=int, help="World model hidden size.")
    parser.add_argument("--world_model_learning_rate", default=0.0001, type=float, help="World model learning rate.")
    parser.add_argument("--world_model_loss_source", default="real", choices=["real", "imag"], help="Whether to train on loss of real or imaginary rollouts.")
    parser.add_argument("--world_model_prediction_type", default="location", choices=["existence", "class", "location"], help="What the model predicts.")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", default="datasets/stage_2_same_worlds_dataset.pickle", help="path to the dataset file")
      
    # Training arguments
    parser.add_argument("--max_rollout_length", default=16, type=int, help="Max length of a rollout to train for")
    parser.add_argument("--update_step", default=64, type=int, help="Number of steps before model update")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size of training input")
    parser.add_argument("--max_time", default=1000, type=float, help="max train time in hrs")
    parser.add_argument("--max_step", default=1e6, type=int, help="max training step")

    # Logging arguments
    parser.add_argument('--eval_step', default=32768, type=int, help='number of steps between evaluations')
    parser.add_argument('--eval_length', default=16, type=int, help='number of steps to run evaluation')
    parser.add_argument('--vis_length', default=16, type=int, help='number of steps to visualize')
    parser.add_argument('--entity', type=str, help="entity to log runs to on wandb")

    args = parser.parse_args()
    if args.output is None:
        args.output = f"output/key-{args.world_model_key_type}-{args.world_model_key_dim}_value-{args.world_model_val_type}-{args.world_model_val_dim}_loss-{args.world_model_loss_source}-{args.world_model_prediction_type}_{int(time.time())}"
    if args.world_model_key_type == "oracle":
        args.world_model_key_dim = 17
    if args.world_model_val_type == "oracle":
        args.world_model_val_dim = 5 # 3 mvmt types + avatar_no_message + avatar_with_message

    assert args.eval_length >= args.vis_length
    
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
    
    wandb.init(
        project = "iw_running",
        entity = args.entity,
        group = f"key-{args.world_model_key_type}-{args.world_model_key_dim}_value-{args.world_model_val_type}-{args.world_model_val_dim}_loss-{args.world_model_loss_source}-{args.world_model_prediction_type}",
        name = str(int(time.time()))
    )
    wandb.config.update(args)
    
    train(args)