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
    # eval_dataloaders = {split: DataLoader(dataset, split, args.max_rollout_length, mode="static", batch_size=args.eval_batch_size)}

    # # Dataloaders
    # print("creating dataloaders")
    # dataloaders = {split: DataLoader(dataset, split, "random" if split == train_split else "static", args.batch_size if split == train_split else args.eval_batch_size, args.max_rollout_length) for split in splits}
    # print("finished creating dataloaders")

    # # Analyzers
    # analyzers = {split: {
    #     "real": Analyzer(world_model if split == train_split else eval_world_model, f"{split}_real_", args.max_rollout_length, world_model.relevant_cls_idxs, args.n_frames, args.device),
    #     "imag": Analyzer(world_model if split == train_split else eval_world_model, f"{split}_imag_", args.max_rollout_length, world_model.relevant_cls_idxs, args.n_frames, args.device),
    # } for split in splits}

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

        # push results to train analyzers for eval
        # if step % args.eval_step > (args.eval_step - args.eval_train_length):
        #     analyzers[train_split]["real"].push(*real_results, (manuals, tokens), ground_truths, (new_idxs, cur_idxs), world_model.real_entity_ids, tensor_timesteps)
        #     analyzers[train_split]["imag"].push(*imag_results, (manuals, tokens), ground_truths, (new_idxs, cur_idxs), world_model.imag_entity_ids, tensor_timesteps)

        # log results and run eval on other splits
        # # run eval
        #     with torch.no_grad():
        #         val_old_tensor_grids = val_tensor_grids
        #         val_grids, val_actions, val_manuals, val_ground_truths, (val_new_idxs, val_cur_idxs), val_timesteps = val_same_worlds_dataloader.step()
        #         val_tensor_grids = torch.from_numpy(val_grids).long().to(args.device)
        #         val_tensor_actions = torch.from_numpy(val_actions).long().to(args.device)
        #         val_tensor_timesteps = torch.from_numpy(val_timesteps).long().to(args.device)
        #         val_manuals, val_tokens = encoder.encode(val_manuals)
            
        #         val_real_results = eval_world_model.real_step(val_old_tensor_grids, val_manuals, val_ground_truths, val_tensor_actions, val_tensor_grids, val_cur_idxs)
        #         val_imag_results = eval_world_model.imag_step(val_manuals, val_ground_truths, val_tensor_actions, val_tensor_grids, val_cur_idxs)

        #         # reset states if new rollouts started
        #         eval_world_model.real_state_reset(val_tensor_grids, val_new_idxs)
        #         eval_world_model.imag_state_reset(val_tensor_grids, val_new_idxs)

        #     val_same_worlds_real_analyzer.push(*val_real_results, (val_manuals, val_tokens), val_ground_truths, (val_new_idxs, val_cur_idxs), eval_world_model.real_entity_ids, val_tensor_timesteps)
        #     val_same_worlds_imag_analyzer.push(*val_imag_results, (val_manuals, val_tokens), val_ground_truths, (val_new_idxs, val_cur_idxs), eval_world_model.imag_entity_ids, val_tensor_timesteps)
        # if step % args.eval_step == 0:
        #     # val_real_loss = eval_world_model.real_loss_reset()
        #     # val_imag_loss = eval_world_model.imag_loss_reset()
        #     eval_log = {
        #         "step": step,
        #         # "val_real_loss": val_real_loss,
        #         # "val_imag_loss": val_imag_loss,
        #         # "val_real_loss_perplexity": np.exp(val_real_loss),
        #         # "val_imag_loss_perplexity": np.exp(val_imag_loss),
        #     }
        #     eval_log.update(analyzers[train_split]["real"].getLog(step))
        #     eval_log.update(analyzers[train_split["imag"]].getLog(step))
        #     # eval_log.update(val_same_worlds_real_analyzer.getLog(step))
        #     # eval_log.update(val_same_worlds_imag_analyzer.getLog(step))
        #     wandb.log(eval_log)

        #     analyzers[train_split]["real"].reset()
        #     analyzers[train_split["imag"]].reset()
            # val_same_worlds_real_analyzer.reset()
            # val_same_worlds_imag_analyzer.reset()

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
    parser.add_argument("--world_model_val_type", default="oracle", choices=["oracle", "emma", "emma-mlp_scale"], help="What to use to process the descriptors' value tokens.")
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
    parser.add_argument('--eval_max_length', default=4096, type=int, help='max number of steps to run evaluation on splits')
    parser.add_argument('--eval_batch_size', default=256, type=int, help='batch_size of unseen split input')
    parser.add_argument('--n_frames', default=32, type=int, help='number of frames to visualize')
    parser.add_argument('--entity', type=str, help="entity to log runs to on wandb")

    args = parser.parse_args()
    if args.output is None:
        args.output = f"output/key-{args.world_model_key_type}-{args.world_model_key_dim}_value-{args.world_model_val_type}-{args.world_model_val_dim}_loss-{args.world_model_loss_source}-{args.world_model_prediction_type}_{int(time.time())}"
    if args.world_model_key_type == "oracle":
        args.world_model_key_dim = 17
    if args.world_model_val_type == "oracle":
        args.world_model_val_dim = 7 # avatar mvmt type + 3 entity mvmt types + 3 entity role types

    assert args.eval_step % args.update_step == 0
    assert args.eval_max_length >= args.n_frames
    
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