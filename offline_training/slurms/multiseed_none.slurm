#!/bin/bash
#SBATCH --job-name=messenger_none
#SBATCH --output=output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:2

source ~/.bashrc
conda activate messenger

home_dir=/n/fs/nlp-kn5378/code/messenger-emma/offline_training

cd $home_dir

for seed in 453 298 237 954 932; do
        echo $seed
        srun --gres=gpu:1 -n 1 --mem=24G --exclusive \
                python3 -u train_khanh.py \
                        --dataset_path custom_dataset/dataset_all_intentions_50k_train.pickle \
                        --exp_name none_50k_hidden_1024_seed_${seed} \
                        --manuals none \
                        --use_wandb 1 \
                        --device 0 \
                        --seed ${seed} &
done

wait;

exit 0
