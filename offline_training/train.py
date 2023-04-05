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
from evaluator import Evaluator

def train(args):
    world_model = BatchedWorldModel(
        key_type=args.world_model_key_type,
        key_dim=args.world_model_key_dim,
        val_type=args.world_model_val_type,
        val_dim=args.world_model_val_dim,
        memory_type=args.world_model_memory_type,
        latent_size=args.world_model_latent_size,
        hidden_size=args.world_model_hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.world_model_learning_rate,
        reward_loss_weight=args.world_model_reward_loss_weight,
        done_loss_weight=args.world_model_done_loss_weight,
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

    # Text Encoder
    encoder_model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = BatchedEncoder(model=encoder_model, tokenizer=tokenizer, device=args.device, max_length=36)

    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)

    splits = list(dataset["rollouts"].keys())
    train_split = None
    for split in splits:
        if "train" in split:
            train_split = split 
            break 
    assert train_split is not None

    train_dataloader = DataLoader(dataset, train_split, args.max_rollout_length, mode="random", batch_size=args.batch_size)
    eval_dataloaders = {split: DataLoader(dataset, split, args.max_rollout_length, mode="static", batch_size=args.eval_batch_size) for split in splits}

    # training variables
    step = 0
    start_time = time.time()

    manuals, ground_truths, grids = train_dataloader.reset()
    manuals, tokens = encoder.encode(manuals)
    tensor_grids = torch.from_numpy(grids).long().to(args.device)
    world_model.real_state_reset(tensor_grids)
    world_model.imag_state_reset(tensor_grids)

    pbar = tqdm(total=args.max_step)
    while step < args.max_step: # main training loop
        if args.world_model_key_freeze and ((args.world_model_key_unfreeze_step >= 0) and (step >= args.world_model_key_unfreeze_step)):
            world_model.unfreeze_key()
            args.world_model_key_freeze = False
        if args.world_model_val_freeze and ((args.world_model_val_unfreeze_step >= 0) and (step >= args.world_model_val_unfreeze_step)):
            world_model.unfreeze_val()
            args.world_model_val_freeze = False

        old_tensor_grids = tensor_grids
        manuals, ground_truths, actions, grids, rewards, dones, (new_idxs, cur_idxs), timesteps = train_dataloader.step()
        manuals, tokens = encoder.encode(manuals)
        tensor_actions = torch.from_numpy(actions).long().to(args.device)
        tensor_grids = torch.from_numpy(grids).long().to(args.device)
        tensor_rewards = torch.from_numpy(rewards).float().to(args.device)
        tensor_dones = torch.from_numpy(dones).long().to(args.device)
        tensor_timesteps = torch.from_numpy(timesteps).long().to(args.device)
    
        # accumulate gradient
        if args.world_model_loss_source == "real":
            real_results = world_model.real_step(old_tensor_grids, manuals, ground_truths, tensor_actions, tensor_grids, tensor_rewards, tensor_dones, cur_idxs)
            with torch.no_grad():
                imag_results = world_model.imag_step(manuals, ground_truths, tensor_actions, tensor_grids, tensor_rewards, tensor_dones, cur_idxs)
        elif args.world_model_loss_source == "imag":
            imag_results = world_model.imag_step(manuals, ground_truths, tensor_actions, tensor_grids, tensor_rewards, tensor_dones, cur_idxs)
            with torch.no_grad():
                real_results = world_model.real_step(old_tensor_grids, manuals, ground_truths, tensor_actions, tensor_grids, tensor_rewards, tensor_dones, cur_idxs)
        else:
            raise NotImplementedError
        step += 1    

        # perform update
        if step % args.update_step == 0:
            if args.world_model_loss_source == "real":
                world_model.real_loss_update()
            elif args.world_model_loss_source == "imag":
                world_model.imag_loss_update()
            real_grid_loss, real_reward_loss, real_done_loss, real_loss = world_model.real_loss_reset()
            imag_grid_loss, imag_reward_loss, imag_done_loss, imag_loss = world_model.imag_loss_reset()
            world_model.real_state_detach()
            world_model.imag_state_detach()
            
            updatelog = {
                "step": step,
                "real_grid_loss": real_grid_loss,
                "real_reward_loss": real_reward_loss,
                "real_done_loss": real_done_loss,
                "real_loss": real_loss,
                "imag_grid_loss": imag_grid_loss,
                "imag_reward_loss": imag_reward_loss,
                "imag_done_loss": imag_done_loss,
                "imag_loss": imag_loss,
                "real_grid_loss_perplexity": np.exp(real_grid_loss),
                "imag_grid_loss_perplexity": np.exp(imag_grid_loss),
                "real_done_loss_perplexity": np.exp(real_done_loss),
                "imag_done_loss_perplexity": np.exp(imag_done_loss),
            }
            wandb.log(updatelog)

        # reset states if new rollouts started
        world_model.real_state_reset(tensor_grids, new_idxs)
        world_model.imag_state_reset(tensor_grids, new_idxs)

        if step % args.eval_step == 0:
            with torch.no_grad():
                eval_updatelog = {"step": step}
                for eval_split, eval_dataloader in eval_dataloaders.items():
                    eval_world_model = BatchedWorldModel(
                        key_type=args.world_model_key_type,
                        key_dim=args.world_model_key_dim,
                        val_type=args.world_model_val_type,
                        val_dim=args.world_model_val_dim,
                        memory_type=args.world_model_memory_type,
                        latent_size=args.world_model_latent_size,
                        hidden_size=args.world_model_hidden_size,
                        batch_size=args.eval_batch_size,
                        learning_rate=args.world_model_learning_rate,
                        reward_loss_weight=args.world_model_reward_loss_weight,
                        done_loss_weight=args.world_model_done_loss_weight,
                        prediction_type=args.world_model_prediction_type,
                        pred_multilabel_threshold=args.world_model_pred_multilabel_threshold,
                        refine_pred_multilabel=args.world_model_refine_pred_multilabel,
                        device=args.device
                    )
                    eval_world_model.load_state_dict(world_model.state_dict())

                    eval_real_evaluator = Evaluator(eval_world_model, f"eval_{split}_real_", args.max_rollout_length, eval_world_model.relevant_cls_idxs, args.n_frames, args.device)
                    eval_imag_evaluator = Evaluator(eval_world_model, f"eval_{split}_imag_", args.max_rollout_length, eval_world_model.relevant_cls_idxs, args.n_frames, args.device)
                
                    eval_manuals, eval_ground_truths, eval_grids, eval_n_rollouts = eval_dataloader.reset()
                    eval_manuals, eval_tokens = encoder.encode(eval_manuals)
                    eval_tensor_grids = torch.from_numpy(eval_grids).long().to(args.device)
                    eval_world_model.real_state_reset(eval_tensor_grids)
                    eval_world_model.imag_state_reset(eval_tensor_grids)

                    eval_pbar = tqdm(total=eval_n_rollouts)
                    while True:
                        eval_old_tensor_grids = eval_tensor_grids
                        eval_manuals, eval_ground_truths, eval_actions, eval_grids, eval_rewards, eval_dones, (eval_new_idxs, eval_cur_idxs), eval_timesteps, eval_just_completes, eval_all_complete = eval_dataloader.step()
                        if eval_all_complete:
                            break
                        eval_manuals, eval_tokens = encoder.encode(eval_manuals)
                        eval_tensor_actions = torch.from_numpy(eval_actions).long().to(args.device)
                        eval_tensor_grids = torch.from_numpy(eval_grids).long().to(args.device)
                        eval_tensor_rewards = torch.from_numpy(eval_rewards).float().to(args.device)
                        eval_tensor_dones = torch.from_numpy(eval_dones).long().to(args.device)
                        eval_tensor_timesteps = torch.from_numpy(eval_timesteps).long().to(args.device)
                    
                        eval_real_results = eval_world_model.real_step(eval_old_tensor_grids, eval_manuals, eval_ground_truths, eval_tensor_actions, eval_tensor_grids, eval_tensor_rewards, eval_tensor_dones, eval_cur_idxs)
                        eval_imag_results = eval_world_model.imag_step(eval_manuals, eval_ground_truths, eval_tensor_actions, eval_tensor_grids, eval_tensor_rewards, eval_tensor_dones, eval_cur_idxs)

                        eval_real_evaluator.push(eval_real_results, (eval_manuals, eval_tokens), eval_ground_truths, (eval_new_idxs, eval_cur_idxs), eval_world_model.real_entity_ids, eval_tensor_timesteps)
                        eval_imag_evaluator.push(eval_imag_results, (eval_manuals, eval_tokens), eval_ground_truths, (eval_new_idxs, eval_cur_idxs), eval_world_model.imag_entity_ids, eval_tensor_timesteps)

                        eval_world_model.real_state_reset(eval_tensor_grids, eval_new_idxs)
                        eval_world_model.imag_state_reset(eval_tensor_grids, eval_new_idxs)

                        eval_pbar.update(int(eval_just_completes.sum().item()))

                    eval_updatelog.update(eval_real_evaluator.getLog(step))
                    eval_updatelog.update(eval_imag_evaluator.getLog(step))
                    eval_real_grid_loss, eval_real_reward_loss, eval_real_done_loss, eval_real_loss = eval_world_model.real_loss_reset()
                    eval_imag_grid_loss, eval_imag_reward_loss, eval_imag_done_loss, eval_imag_loss = eval_world_model.imag_loss_reset()
                    eval_updatelog.update({
                        f"eval_{eval_split}_real_grid_loss": eval_real_grid_loss,
                        f"eval_{eval_split}_real_reward_loss": eval_real_reward_loss,
                        f"eval_{eval_split}_real_done_loss": eval_real_done_loss,
                        f"eval_{eval_split}_real_loss": eval_real_loss,
                        f"eval_{eval_split}_imag_grid_loss": eval_imag_grid_loss,
                        f"eval_{eval_split}_imag_reward_loss": eval_imag_reward_loss,
                        f"eval_{eval_split}_imag_done_loss": eval_imag_done_loss,
                        f"eval_{eval_split}_imag_loss": eval_imag_loss,
                        f"eval_{eval_split}_real_grid_loss_perplexity": np.exp(eval_real_grid_loss),
                        f"eval_{eval_split}_imag_grid_loss_perplexity": np.exp(eval_imag_grid_loss),
                        f"eval_{eval_split}_real_done_loss_perplexity": np.exp(eval_real_done_loss),
                        f"eval_{eval_split}_imag_done_loss_perplexity": np.exp(eval_imag_done_loss),
                    })
                wandb.log(eval_updatelog)

        pbar.update(1)
        # check if max_time has elapsed
        if time.time() - start_time > 60 * 60 * args.max_time:
            break
    pbar.close()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output", default=None, type=str, help="Local output file name or path.")
    parser.add_argument("--seed", default=0, type=int, help="Set the seed for the model and training.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")

    # World model arguments
    parser.add_argument("--world_model_pred_multilabel_threshold", default=0.5, type=float, help="Probability threshold to predict existence of a sprite in pred_multilabel when prediction_type == 'existence'.")
    parser.add_argument("--world_model_refine_pred_multilabel", default=True, action="store_true", help="Refine pred_multilabel by keeping <=1 of each sprite, exactly 1 avatar, and no entities known to not exist in the level.")
    parser.add_argument("--world_model_no_refine_pred_multilabel", dest="world_model_refine_pred_multilabel", action="store_false", help="Do not refine pred_multilabel.")
    parser.add_argument("--world_model_load_state", default=None, help="Path to world model state dict.")
    parser.add_argument("--world_model_key_dim", default=256, type=int, help="World model key embedding dimension.")
    parser.add_argument("--world_model_key_type", default="oracle", choices=["oracle", "emma", "emma-mlp_scale", "chatgpt"], help="What to use to process the descriptors' key tokens.")
    parser.add_argument("--world_model_key_freeze", default=False, action="store_true", help="Whether to freeze key module.")
    parser.add_argument("--world_model_key_unfreeze_step", default=5e5, type=int, help="Train step to unfreeze key module, -1 means never.")
    parser.add_argument("--world_model_val_dim", default=256, type=int, help="World model value embedding dimension.")
    parser.add_argument("--world_model_val_type", default="oracle", choices=["oracle", "emma", "emma-mlp_scale", "chatgpt", "none"], help="What to use to process the descriptors' value tokens.")
    parser.add_argument("--world_model_val_freeze", default=False, action="store_true", help="Whether to freeze val module.")
    parser.add_argument("--world_model_val_unfreeze_step", default=5e5, type=int, help="Train step to unfreeze val module, -1 means never.")
    parser.add_argument("--world_model_latent_size", default=512, type=int, help="World model latent size.")
    parser.add_argument("--world_model_hidden_size", default=1024, type=int, help="World model hidden size.")
    parser.add_argument("--world_model_learning_rate", default=0.0005, type=float, help="World model learning rate.")
    parser.add_argument("--world_model_reward_loss_weight", default=1, type=float, help="World model reward loss weight.")
    parser.add_argument("--world_model_done_loss_weight", default=1, type=float, help="World model done loss weight.")
    parser.add_argument("--world_model_loss_source", default="real", choices=["real", "imag"], help="Whether to train on loss of real or imaginary rollouts.")
    parser.add_argument("--world_model_prediction_type", default="location", choices=["existence", "class", "location"], help="What the model predicts.")
    parser.add_argument("--world_model_memory_type", default="lstm", choices=["mlp", "lstm"], help="NN type for memory module of world model.")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", default="custom_dataset/dataset.pickle", help="path to the dataset file")
      
    # Training arguments
    parser.add_argument("--max_rollout_length", default=32, type=int, help="Max length of a rollout to train for")
    parser.add_argument("--update_step", default=32, type=int, help="Number of steps before model update")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size of training input")
    parser.add_argument("--max_time", default=1000, type=float, help="max train time in hrs")
    parser.add_argument("--max_step", default=1e7, type=int, help="max training step")

    # Logging arguments
    parser.add_argument('--eval_step', default=32768, type=int, help='number of steps between evaluations')
    parser.add_argument('--eval_batch_size', default=2048, type=int, help='batch_size for evaluation')
    parser.add_argument('--n_frames', default=32, type=int, help='number of frames to visualize')
    parser.add_argument('--entity', type=str, help="entity to log runs to on wandb")

    args = parser.parse_args()
    if args.output is None:
        args.output = f"output/key-{args.world_model_key_type}-{args.world_model_key_dim}_value-{args.world_model_val_type}-{args.world_model_val_dim}_loss-{args.world_model_loss_source}-{args.world_model_prediction_type}_{int(time.time())}"
    if args.world_model_key_type == "oracle":
        args.world_model_key_dim = 17
    if args.world_model_val_type == "oracle":
        args.world_model_val_dim = 7 # avatar mvmt type + 3 entity mvmt types + 3 entity role types
    elif args.world_model_val_type == "none":
        args.world_model_val_dim = 0 # none

    assert args.eval_step % args.update_step == 0
    
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
        project = "paper_running",
        entity = args.entity,
        group = f"key-{args.world_model_key_type}-{args.world_model_key_dim}_value-{args.world_model_val_type}-{args.world_model_val_dim}_loss-{args.world_model_loss_source}-{args.world_model_prediction_type}",
        name = str(int(time.time()))
    )
    wandb.config.update(args)
    
    train(args)