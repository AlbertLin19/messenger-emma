#!/bin/bash
#SBATCH --job-name=messenger_oracle
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:1

for seed in 453 298 237 954 932; do 
  for model in none oracle gpt embed; do
    exp_dir=${model}_50k_hidden_1024_seed_${seed}
    model_path=experiments/${exp_dir}/dev_ne_nr_or_nm_best_total_loss.ckpt
    echo $model_path
    python3 -u train_khanh.py \
          --dataset_path custom_dataset/dataset_all_intentions_50k_train.pickle \
          --exp_name ${exp_dir} \
          --manuals $model \
          --device 0 \
          --load_model_from $model_path \
          --eval_mode 1
  done
done


