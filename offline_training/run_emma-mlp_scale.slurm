#!/bin/bash
#SBATCH --job-name=messenger_emma-mlp_scale
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:1

srun python train.py --world_model_key_type emma-mlp_scale --world_model_key_freeze --world_model_key_unfreeze_step 500000 --world_model_val_type emma-mlp_scale --world_model_val_freeze --world_model_val_unfreeze_step 500000