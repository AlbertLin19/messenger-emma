#!/bin/bash

ENTITY="aklin" # replace with your wandb entity
LOG_GROUP="emma_s1" # group to log runs in wandb
SLURM_ARGS="-A pnlp" # replace with your slurm args

CMD="python ../train.py --splits_path ../../offline_training/custom_dataset/data_splits_final.json --lr 0.00005 --update_timestep 64 --optimizer Adam --weight_decay 0.0 --max_time 11.75 --entity $ENTITY --log_group $LOG_GROUP"

mkdir -p output
mkdir -p slurm
for seed in 1 2 3;
do
    sbatch $SLURM_ARGS stage_1.slurm $CMD --seed $seed --output output/emma_s1_$seed
done