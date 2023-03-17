import pickle
import openai

from tqdm import tqdm
from messenger.envs.config import NPCS
ENTITY_IDS = {entity.name: entity.id for entity in NPCS}
MOVEMENT_TYPES = {
    "chaser": 0,
    "fleeing": 1,
    "immovable": 2,
}

PROMPT_TEMPLATE = """
This is a list of entity names and their corresponding IDs:
airplane: 2
mage: 3
dog: 4
bird: 5
fish: 6
scientist: 7
thief: 8
ship: 9
ball: 10
robot: 11
queen: 12
sword: 13

This is a list of movement names and their corresponding IDs:
chasing: 0
fleeing: 1
stationary: 2

Respond strictly in the following format. Do not add anything else.
entity ID, movement ID

What is the entity ID and movement ID that corresponds to the following description?
%s
"""

DATASET_PATH = "datasets/stage_2_same_worlds_dataset.pickle"
NEW_DATASET_PATH = "datasets/stage_2_same_worlds_dataset_with_chatgpt_groundings.pickle"

with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)

for data_name, data in dataset.items():
    print(data_name)
    manuals = data["manuals"]
    ground_truths = data["ground_truths"]
    chatgpt_groundings = []

    n_entity_total = 0
    n_entity_correct = 0
    n_mvmt_total = 0
    n_mvmt_correct = 0

    for i in tqdm(range(len(manuals))):
        manual = manuals[i]
        ground_truth = ground_truths[i]
        chatgpt_grounding = []

        for j in range(len(manual)):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": PROMPT_TEMPLATE % manual[j]}]
            )["choices"][0]["message"]["content"]
            entity_id, movement_id = (response + ",").replace(" ", "").split(",")[:2]
            entity_id = int(entity_id) if entity_id.isdigit() else -1
            entity_id = entity_id if entity_id in ENTITY_IDS.values() else -1
            movement_id = int(movement_id) if movement_id.isdigit() else -1
            movement_id = movement_id if movement_id in MOVEMENT_TYPES.values() else -1

            if ENTITY_IDS[ground_truth[j][0]] == entity_id:
                n_entity_correct += 1
            if MOVEMENT_TYPES[ground_truth[j][1]] == movement_id:
                n_mvmt_correct += 1
            n_entity_total += 1
            n_mvmt_total += 1

            chatgpt_grounding.append((entity_id, movement_id))

        chatgpt_groundings.append(chatgpt_grounding)

    data["chatgpt_groundings"] = chatgpt_groundings
    print(f"entities correct: {n_entity_correct}/{n_entity_total}")
    print(f"mvmts correct: {n_mvmt_correct}/{n_mvmt_total}")

with open(NEW_DATASET_PATH, "wb") as f:
    pickle.dump(dataset, f)
