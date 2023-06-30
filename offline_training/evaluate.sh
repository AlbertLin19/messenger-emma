#!/bin/bash
#SBATCH --job-name=messenger_oracle
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:1

for model in none oracle gpt embed
do
  model_path=experiments/${model}_paper_10k_train_500_eval/dev_ne_nr_or_nm_best_total_loss.ckpt
  echo $model_path
  python3 -u train_khanh.py \
        --dataset_path custom_dataset/dataset_shuffle_balanced_intentions_10k_train_500_eval.pickle \
        --exp_name ${model}_paper_10k_train_500_eval \
        --manuals $model \
        --device 0 \
        --load_model_from $model_path \
        --eval_mode 1
done


