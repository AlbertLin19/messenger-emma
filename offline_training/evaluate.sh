#!/bin/bash
#SBATCH --job-name=messenger_fun
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:1

n_layer=8                                                                                           
n_embd=256                                                                                         
n_head=4                                                                                            

manual=none
extra=nlayer_${n_layer}_nembd_${n_embd}_nhead_${n_embd}

python3 -u train_transformer.py \
                   --dataset_path custom_dataset/dataset_transformer_10k_train_500_eval.pickle \
                   --description_tokenizer custom_dataset/tokenizer_transformer.json \
                   --transformer_config_file gpt2-small-config.json \
                   --exp_name tmp2 \
                   --manuals ${manual} \
                   --eval_mode 1 \
                   --load_model experiments/fun_transformer_v2_${manual}_${extra}/train_dev_games_best_state_loss.ckpt \
                   --transformer_n_layer ${n_layer} \
                   --transformer_n_embd ${n_embd} \
                   --transformer_n_head ${n_head} \
