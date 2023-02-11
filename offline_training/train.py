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

    eval_world_model = BatchedWorldModel(
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
    eval_world_model.load_state_dict(world_model.state_dict())

    # Analyzers
    train_all_real_analyzer = Analyzer("real_", args.eval_length, args.vis_length, world_model.relevant_cls_idxs)
    train_all_imag_analyzer = Analyzer("imag_", args.eval_length, args.vis_length, world_model.relevant_cls_idxs)
    val_same_worlds_real_analyzer = Analyzer("val_real_", args.eval_length, args.vis_length, world_model.relevant_cls_idxs)
    val_same_worlds_imag_analyzer = Analyzer("val_imag_", args.eval_length, args.vis_length, world_model.relevant_cls_idxs)

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
    world_model.real_state_reset(tensor_grids)
    world_model.imag_state_reset(tensor_grids)

    val_grids, val_actions, val_manuals, val_ground_truths = val_same_worlds_dataloader.reset()
    val_tensor_grids = torch.from_numpy(val_grids).long().to(args.device)
    val_tensor_actions = torch.from_numpy(val_actions).long().to(args.device)
    val_manuals, _ = encoder.encode(val_manuals)
    eval_world_model.real_state_reset(val_tensor_grids)
    eval_world_model.imag_state_reset(val_tensor_grids)

    pbar = tqdm(total=args.max_step)
    while step < args.max_step: # main training loop
        if args.world_model_key_freeze and ((args.world_model_key_unfreeze_step >= 0) and (step >= args.world_model_key_unfreeze_step)):
            world_model.unfreeze_key()
            args.world_model_key_freeze = False
        if args.world_model_val_freeze and ((args.world_model_val_unfreeze_step >= 0) and (step >= args.world_model_val_unfreeze_step)):
            world_model.unfreeze_val()
            args.world_model_val_freeze = False

        old_tensor_grids = tensor_grids
        grids, actions, manuals, ground_truths, (new_idxs, cur_idxs) = train_all_dataloader.step()
        tensor_grids = torch.from_numpy(grids).long().to(args.device)
        tensor_actions = torch.from_numpy(actions).long().to(args.device)
        manuals, _ = encoder.encode(manuals)
    
        # accumulate gradient
        if args.world_model_loss_source == "real":
            real_results = world_model.real_step(old_tensor_grids, manuals, ground_truths, tensor_actions, tensor_grids, cur_idxs)
            with torch.no_grad():
                imag_results = world_model.imag_step(manuals, ground_truths, tensor_actions, tensor_grids, cur_idxs)
        elif args.world_model_loss_source == "imag":
            imag_results = world_model.imag_step(manuals, ground_truths, tensor_actions, tensor_grids, cur_idxs)
            with torch.no_grad():
                real_results = world_model.real_step(old_tensor_grids, manuals, ground_truths, tensor_actions, tensor_grids, cur_idxs)
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

            eval_world_model.load_state_dict(world_model.state_dict())
            
            updatelog = {
                "step": step,
                "real_loss": real_loss,
                "imag_loss": imag_loss,
            }
            wandb.log(updatelog)

        # reset states if new rollouts started
        world_model.real_state_reset(tensor_grids, new_idxs)
        world_model.imag_state_reset(tensor_grids, new_idxs)

        # push results to analyzers and run eval
        if step % args.eval_step > (args.eval_step - args.eval_length):
            train_all_real_analyzer.push(*real_results)
            train_all_imag_analyzer.push(*imag_results)

            # run eval
            with torch.no_grad():
                val_old_tensor_grids = val_tensor_grids
                val_grids, val_actions, val_manuals, val_ground_truths, (val_new_idxs, val_cur_idxs) = val_same_worlds_dataloader.step()
                val_tensor_grids = torch.from_numpy(val_grids).long().to(args.device)
                val_tensor_actions = torch.from_numpy(val_actions).long().to(args.device)
                val_manuals, _ = encoder.encode(val_manuals)
            
                val_real_results = eval_world_model.real_step(val_old_tensor_grids, val_manuals, val_ground_truths, val_tensor_actions, val_tensor_grids, val_cur_idxs)
                val_imag_results = eval_world_model.imag_step(val_manuals, val_ground_truths, val_tensor_actions, val_tensor_grids, val_cur_idxs)

                # reset states if new rollouts started
                eval_world_model.real_state_reset(val_tensor_grids, val_new_idxs)
                eval_world_model.imag_state_reset(val_tensor_grids, val_new_idxs)

            val_same_worlds_real_analyzer.push(*val_real_results)
            val_same_worlds_imag_analyzer.push(*val_imag_results)

        # log results
        if step % args.eval_step == 0:
            val_real_loss = eval_world_model.real_loss_reset()
            val_imag_loss = eval_world_model.imag_loss_reset()
            eval_log = {
                "step": step,
                "val_real_loss": val_real_loss,
                "val_imag_loss": val_imag_loss,
            }
            eval_log.update(train_all_real_analyzer.getLog())
            eval_log.update(train_all_imag_analyzer.getLog())
            eval_log.update(val_same_worlds_real_analyzer.getLog())
            eval_log.update(val_same_worlds_imag_analyzer.getLog())
            wandb.log(eval_log)

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
    parser.add_argument("--world_model_learning_rate", default=0.0005, type=float, help="World model learning rate.")
    parser.add_argument("--world_model_loss_source", default="real", choices=["real", "imag"], help="Whether to train on loss of real or imaginary rollouts.")
    parser.add_argument("--world_model_prediction_type", default="location", choices=["existence", "class", "location"], help="What the model predicts.")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", default="datasets/stage_2_same_worlds_dataset.pickle", help="path to the dataset file")
      
    # Training arguments
    parser.add_argument("--max_rollout_length", default=32, type=int, help="Max length of a rollout to train for")
    parser.add_argument("--update_step", default=32, type=int, help="Number of steps before model update")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size of training input")
    parser.add_argument("--max_time", default=1000, type=float, help="max train time in hrs")
    parser.add_argument("--max_step", default=1e7, type=int, help="max training step")

    # Logging arguments
    parser.add_argument('--eval_step', default=32768, type=int, help='number of steps between evaluations')
    parser.add_argument('--eval_length', default=32, type=int, help='number of steps to run evaluation')
    parser.add_argument('--vis_length', default=32, type=int, help='number of steps to visualize')
    parser.add_argument('--entity', type=str, help="entity to log runs to on wandb")

    args = parser.parse_args()
    if args.output is None:
        args.output = f"output/key-{args.world_model_key_type}-{args.world_model_key_dim}_value-{args.world_model_val_type}-{args.world_model_val_dim}_loss-{args.world_model_loss_source}-{args.world_model_prediction_type}_{int(time.time())}"
    if args.world_model_key_type == "oracle":
        args.world_model_key_dim = 17
    if args.world_model_val_type == "oracle":
        args.world_model_val_dim = 4 # 3 entity mvmt types + avatar mvmt type

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