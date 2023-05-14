#!/bin/bash
#SBATCH --job-name=messenger_embed
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:1

python3 -u train_khanh.py \
                   --dataset_path custom_dataset/dataset_shuffle_balanced_intentions_10k_train_500_eval.pickle \
                   --exp_name embed_loc_loss_only_shuffle_balanced_intentions_10k_train_500_eval \
                   --manuals embed \
                   --use_true_attention 0 \
                   --train_loc_loss_only 1 \
                   --device 1 \
                   --use_wandb 1
