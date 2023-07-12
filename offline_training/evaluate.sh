#!/bin/bash
#SBATCH --job-name=messenger_fun
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:1

python3 -u train_transformer.py \
                   --dataset_path custom_dataset/dataset_transformer_10k_train_500_eval.pickle \
                   --description_tokenizer custom_dataset/tokenizer_transformer.json \
                   --transformer_config_file gpt2-small-config.json \
                   --exp_name tmp2 \
                   --manuals oracle \
                   --eval_mode 1 \
                   --load_model experiments/fun_transformer_v2_oracle/dev_ne_nr_or_nm_best_state_loss.ckpt
