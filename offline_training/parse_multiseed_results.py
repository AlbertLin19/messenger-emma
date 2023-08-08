import os
import sys
from collections import defaultdict
import numpy as np

all_models = ['none', 'embed', 'gpt', 'oracle']
all_splits = ['train_games', 'test_se_nr_or_nm', 'test_ne_sr_and_sm', 'test_ne_nr_or_nm']
all_metrics = ['total_loss', 'loc_loss', 'id_loss', 'reward_loss', 'done_loss']

results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

model = None
split = None

with open("multiseed_results.log") as f:
    for line in f:
        if "Loaded model from" in line:
            for m in all_models:
                if m in line:
                    model = m
                    break
        if "EVALUATION" in line:
            for s in all_splits:
                if s in line:
                    split = s
                    break
        if "BEST" in line:
            items = line[line.find("BEST") + 4:].strip().split(',')
            for x in items:
                x = x.replace(",", "")
                for metric in all_metrics:
                    if metric in x:
                        value = float(x.split()[-1])
                        results[model][split][metric].append(value)

for split in all_splits:
    print(split)
    for metric in all_metrics:
        print(metric)
        for model in all_models:
            print(model, end='\t')
            for value in results[model][split][metric]:
                print(value, end='\t')
            print()
