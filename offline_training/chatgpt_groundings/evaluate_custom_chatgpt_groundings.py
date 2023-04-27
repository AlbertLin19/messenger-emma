import json

ENTITY_IDS = {
    "robot": 11,
    "airplane": 2,
    "thief": 8,
    "scientist": 7,
    "queen": 12,
    "ship": 9,
    "dog": 4,
    "bird": 5,
    "fish": 6,
    "mage": 3,
    "orb": 10,
    "sword": 13,
}
MOVEMENT_TYPES = {
    "chasing": 0,
    "fleeing": 1,
    "immobile": 2,
}
ROLE_TYPES = {
    "enemy": 0,
    "message": 1,
    "goal": 2,
}

TEXT_FILE = "../../messenger/envs/texts/custom_text_splits/custom_text_splits.json"
CHATGPT_GROUNDINGS_FILE = "../../messenger/envs/texts/chatgpt_groundings/chatgpt_grounding_for_text_all.json"

with open(TEXT_FILE, "r") as f:
    texts = json.load(f)
with open(CHATGPT_GROUNDINGS_FILE, "r") as f:
    chatgpt_groundings = json.load(f)

splits = list(list(list(list(texts.values())[0].values())[0].values())[0].keys())

for split in splits:
    print(split)

    n_entity_correct = 0
    n_entity_incorrect = 0
    n_entity_null = 0

    n_mvmt_correct = 0
    n_mvmt_incorrect = 0
    n_mvmt_null = 0

    n_role_correct = 0
    n_role_incorrect = 0
    n_role_null = 0

    for entity_type, entity_texts in texts.items():
        entity_id_gt = ENTITY_IDS[entity_type]
        for movement_type, movement_texts in entity_texts.items():
            movement_id_gt = MOVEMENT_TYPES[movement_type]
            for role_type, role_texts in movement_texts.items():
                role_id_gt = ROLE_TYPES[role_type]
                for split_type, split_texts in role_texts.items():
                    if split_type != split:
                        continue
                    for text in split_texts:
                        entity_id, movement_id, role_id = chatgpt_groundings[text]

                        if entity_id == entity_id_gt:
                            n_entity_correct += 1
                        elif entity_id not in ENTITY_IDS.values():
                            n_entity_null += 1
                        else:
                            n_entity_incorrect += 1

                        if movement_id == movement_id_gt:
                            n_mvmt_correct += 1
                        elif movement_id not in MOVEMENT_TYPES.values():
                            n_mvmt_null += 1
                        else:
                            n_mvmt_incorrect += 1

                        if role_id == role_id_gt:
                            n_role_correct += 1
                        elif role_id not in ROLE_TYPES.values():
                            n_role_null += 1
                        else:
                            n_role_incorrect += 1

    total = n_entity_correct + n_entity_incorrect + n_entity_null
    print(f"total: {total}")
    print(f"entities  correct | incorrect | null: {n_entity_correct/total:.4f} | {n_entity_incorrect/total:.4f} | {n_entity_null/total:.4f}")
    print(f"movements correct | incorrect | null: {n_mvmt_correct/total:.4f} | {n_mvmt_incorrect/total:.4f} | {n_mvmt_null/total:.4f}")
    print(f"roles     correct | incorrect | null: {n_role_correct/total:.4f} | {n_role_incorrect/total:.4f} | {n_role_null/total:.4f}")
    print()