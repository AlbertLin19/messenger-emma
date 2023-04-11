#!/bin/bash
#SBATCH --job-name=messenger_none
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:1

srun python train.py --dataset_path custom_dataset/dataset_32x.pickle --world_model_val_type none