import json 
import pickle 
from tqdm import tqdm

SAVE_PATH = "./dataset.pickle"
SPLITS_PATH = "./splits.json"
GAME_REPEATS = 32

with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

with open(SAVE_PATH, 'rb') as f:
    dataset = pickle.load(f)

keys = dataset["keys"]

for split, games in splits.items():
    print("verifying", split)
    violation = False
    for game in tqdm(games):
        count = 0
        for ground_truth_idx in dataset["rollouts"][split]["ground_truth_idxs"]:
            match = True
            for i in range(len(game)):
                if game[i][0] != keys["entities"][ground_truth_idx[i][0]]:
                    match = False 
                    break
                if game[i][1] != keys["dynamics"][ground_truth_idx[i][1]]:
                    match = False
                    break
                if game[i][2] != keys["roles"][ground_truth_idx[i][2]]:
                    match = False
                    break
            if match:
                count += 1
        if count != GAME_REPEATS:
            violation = True
            print(f"violation: count {count} != GAME_REPEATS {GAME_REPEATS}")
    if not violation:
        print("no violation")
