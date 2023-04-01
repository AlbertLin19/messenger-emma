import os
import json

from messenger.envs.config import NPCS
ENTITY_IDS = {entity.name: entity.id for entity in NPCS}
MOVEMENT_TYPES = {
    "chaser": 0,
    "fleeing": 1,
    "immovable": 2,
}

TEXT_DIR = "../../messenger/envs/texts"
SAVE_DIR = "../../messenger/envs/texts/chatgpt_groundings"
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
    
    chatgpt_grounding_paths = [os.path.join(SAVE_DIR, "chatgpt_grounding_for_" + text_file)]
    while os.path.isfile(os.path.join(SAVE_DIR, f"refined_{len(chatgpt_grounding_paths)}_chatgpt_grounding_for_" + text_file)):
        chatgpt_grounding_paths.append(os.path.join(SAVE_DIR, f"refined_{len(chatgpt_grounding_paths)}_chatgpt_grounding_for_" + text_file))
    for i in range(len(chatgpt_grounding_paths)):
        with open(chatgpt_grounding_paths[i], "r") as f:
            chatgpt_grounding = json.load(f)
            if len(chatgpt_grounding.keys()) == len(text_tuples):
                print(f"version {i}")  
                n_entity_correct = 0
                n_mvmt_correct = 0
                for i in range(len(text_tuples)):
                    entity_id_gt, movement_id_gt, text = text_tuples[i]
                    entity_id, movement_id = chatgpt_grounding[text]
                    if entity_id == entity_id_gt:
                        n_entity_correct += 1
                    if movement_id == movement_id_gt:
                        n_mvmt_correct += 1
                print(f"entities correct: {n_entity_correct}/{len(text_tuples)} = {n_entity_correct/len(text_tuples)}")
                print(f"mvmts correct: {n_mvmt_correct}/{len(text_tuples)} = {n_mvmt_correct/len(text_tuples)}")
                print()
    print()