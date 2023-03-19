import os
import json
import time
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
This is a list of entities and their corresponding IDs:
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

This is a list of movement types and their corresponding IDs:
chasing: 0
fleeing: 1
stationary: 2

Respond strictly in the following format. Do not add anything else.
entity ID, movement ID

For example, an appropriately formatted response is:
2, 0

Respond with the best choices from the provided lists even if they are not perfect answers. What is the entity ID and movement ID that best correspond to the following description?
%s
"""

TEXT_DIR = "../messenger/envs/texts"
TEXT_FILES = ["text_train.json", "text_val.json", "text_test.json"]

for text_file in TEXT_FILES:
    print(text_file)

    text_tuples = []
    with open(os.path.join(TEXT_DIR, text_file), "r") as f:
        texts = json.load(f)
        
        for entity_type, entity_texts in texts.items():
            if entity_type not in ENTITY_IDS.keys():
                continue
            for role_type, role_texts in entity_texts.items():
                for movement_type, movement_texts in role_texts.items():
                    if movement_type not in MOVEMENT_TYPES.keys():
                        continue
                    for movement_text in movement_texts:
                        text_tuples.append((ENTITY_IDS[entity_type], MOVEMENT_TYPES[movement_type], movement_text))
    
    version = 0
    chatgpt_grounding_path = os.path.join(TEXT_DIR, "chatgpt_grounding_for_" + text_file)
    while os.path.isfile(os.path.join(TEXT_DIR, f"refined_{version+1}_chatgpt_grounding_for_" + text_file)):
        chatgpt_grounding_path = os.path.join(TEXT_DIR, f"refined_{version+1}_chatgpt_grounding_for_" + text_file)
        version += 1
    if os.path.isfile(chatgpt_grounding_path):
        with open(chatgpt_grounding_path, "r") as f:
            chatgpt_grounding = json.load(f)
            if len(chatgpt_grounding.keys()) == len(text_tuples):
                chatgpt_grounding_path = os.path.join(TEXT_DIR, f"refined_{version+1}_chatgpt_grounding_for_" + text_file)
                version += 1
                old_chatgpt_grounding = chatgpt_grounding
                chatgpt_grounding = {}
            else:
                if version > 1:
                    old_chatgpt_grounding_path = os.path.join(TEXT_DIR, f"refined_{version-1}_chatgpt_grounding_for_" + text_file)
                else:
                    old_chatgpt_grounding_path = os.path.join(TEXT_DIR, "chatgpt_grounding_for_" + text_file)
                with open(old_chatgpt_grounding_path, "r") as g:
                    old_chatgpt_grounding = json.load(g)
    else:
        chatgpt_grounding = {}
    print(f"version {version}")
    
    n_entity_correct = 0
    n_mvmt_correct = 0
    for i in tqdm(range(len(text_tuples))):
        entity_id_gt, movement_id_gt, text = text_tuples[i]

        if text in chatgpt_grounding:
            entity_id, movement_id = chatgpt_grounding[text]

        elif version > 0 and old_chatgpt_grounding[text][0] >= 0 and old_chatgpt_grounding[text][1] >= 0:
            entity_id, movement_id = old_chatgpt_grounding[text]
            chatgpt_grounding[text] = (entity_id, movement_id)
            with open(chatgpt_grounding_path, "w") as f:
                json.dump(chatgpt_grounding, f)
            
        else:
            time.sleep(0.2)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": PROMPT_TEMPLATE % text}]
            )["choices"][0]["message"]["content"]
            entity_id, movement_id = (response.strip() + ",").replace(" ", "").split(",")[:2]
            entity_id = int(entity_id) if entity_id.isdigit() else -1
            entity_id = entity_id if entity_id in ENTITY_IDS.values() else -1
            movement_id = int(movement_id) if movement_id.isdigit() else -1
            movement_id = movement_id if movement_id in MOVEMENT_TYPES.values() else -1
            chatgpt_grounding[text] = (entity_id, movement_id)

            if entity_id < 0 or movement_id < 0:
                print(response)

            with open(chatgpt_grounding_path, "w") as f:
                json.dump(chatgpt_grounding, f)

        if entity_id == entity_id_gt:
            n_entity_correct += 1
        if movement_id == movement_id_gt:
            n_mvmt_correct += 1

    print(f"entities correct: {n_entity_correct}/{len(text_tuples)}")
    print(f"mvmts correct: {n_mvmt_correct}/{len(text_tuples)}")