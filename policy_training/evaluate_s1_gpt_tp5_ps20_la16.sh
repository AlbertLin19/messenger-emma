#!/bin/bash
#SBATCH --job-name=evaluate_gpt
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:1

python3 -u evaluate.py \
    --load_state stage_1/output/emma_s1_1_dev_ne_nr_or_nm_max.pth \
    --world_model_manuals gpt \
    --world_model_load_model_from ../offline_training/experiments/gpt_shuffle_balanced_intentions_10k_train_500_eval/dev_ne_nr_or_nm_best_total_loss.ckpt \
    --policy_temperature 5 \
    --num_policy_samples 20 \
    --max_lookahead_length 16