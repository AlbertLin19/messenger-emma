import os
import json
import random
import numpy as np

ENTITIES = [
    "robot",
    "airplane",
    "thief",
    "scientist",
    "queen",
    "ship",
    "dog",
    "bird",
    "fish",
    "mage",
    "orb",
    "sword",
]

MESSENGER_ENTITIES = [
    "robot",
    "airplane",
    "thief",
    "scientist",
    "queen",
    "ship",
    "dog",
    "bird",
    "fish",
    "mage",
    "ball",
    "sword",
]

DYNAMICS = ["chasing", "fleeing", "immobile"]

MESSENGER_DYNAMICS = ["chaser", "fleeing", "immovable"]

ROLES = ["message", "enemy", "goal"]

TEXT_DIR = "../../messenger/envs/texts"
SAVE_DIR = "../../messenger/envs/texts/custom_text_splits"
TEXT_FILES = ["text_train.json", "text_val.json", "text_test.json"]
SPLITS_PATH = "../custom_dataset/data_splits_final_with_test.json"

with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)
for split, games in splits.items():
    print(split, "with", len(games), "games")

SPLIT_NAMES = list(splits.keys())

# count the occurences of entity-dynamic-role combos in splits
split_counts = np.zeros(
    (len(ENTITIES), len(DYNAMICS), len(ROLES), len(SPLIT_NAMES)), dtype=int
)
for split, games in splits.items():
    for game in games:
        for entity in game:
            split_counts[
                ENTITIES.index(entity[0]),
                DYNAMICS.index(entity[1]),
                ROLES.index(entity[2]),
                SPLIT_NAMES.index(split),
            ] += 1
for i in range(len(ENTITIES)):
    print(ENTITIES[i])
    for j in range(len(DYNAMICS)):
        print("    " + DYNAMICS[j])
        for k in range(len(ROLES)):
            print("        " + ROLES[k])
            print(
                "            "
                + "".join(
                    SPLIT_NAMES[l] + ": " + str(split_counts[i, j, k, l]) + "  "
                    for l in range(len(SPLIT_NAMES))
                )
            )

# aggregate all texts
with open(os.path.join(TEXT_DIR, TEXT_FILES[0]), "r") as f:
    all_texts = json.load(f)
for i in range(1, len(TEXT_FILES)):
    with open(os.path.join(TEXT_DIR, TEXT_FILES[i]), "r") as f:
        new_texts = json.load(f)
        for entity, entity_descriptors in new_texts.items():
            for role, role_descriptors in entity_descriptors.items():
                for dynamic, dynamic_descriptors in role_descriptors.items():
                    all_texts[entity][role][dynamic].extend(dynamic_descriptors)

# count the occurences of entity-dynamic-role combos in texts
descriptor_counts = np.zeros(
    (len(MESSENGER_ENTITIES), len(MESSENGER_DYNAMICS), len(ROLES)), dtype=int
)
for entity, entity_descriptors in all_texts.items():
    for role, role_descriptors in entity_descriptors.items():
        for dynamic, dynamic_descriptors in role_descriptors.items():
            if dynamic != "unknown":
                descriptor_counts[
                    MESSENGER_ENTITIES.index(entity),
                    MESSENGER_DYNAMICS.index(dynamic),
                    ROLES.index(role),
                ] += len(dynamic_descriptors)
for i in range(len(MESSENGER_ENTITIES)):
    print(MESSENGER_ENTITIES[i])
    for j in range(len(MESSENGER_DYNAMICS)):
        print("    " + MESSENGER_DYNAMICS[j])
        for k in range(len(ROLES)):
            print("        " + ROLES[k] + ": " + str(descriptor_counts[i, j, k]))
print(np.sum(descriptor_counts), "total")

# distribute descriptors proportionally to splits (but first ensure every split has >=1 of what they need)
base_counts = split_counts != 0
proportional_counts = (
    base_counts
    + np.rint(
        (split_counts / split_counts.sum(axis=-1, keepdims=True))
        * (descriptor_counts[..., None] - base_counts.sum(axis=-1, keepdims=True))
    )
).astype(int)
assert np.all(proportional_counts[split_counts != 0])
assert np.all(proportional_counts[split_counts == 0] == 0)
for i in range(len(ENTITIES)):
    print(ENTITIES[i])
    for j in range(len(DYNAMICS)):
        print("    " + DYNAMICS[j])
        for k in range(len(ROLES)):
            print("        " + ROLES[k])
            print(
                "            "
                + "".join(
                    SPLIT_NAMES[l] + ": " + str(proportional_counts[i, j, k, l]) + "  "
                    for l in range(len(SPLIT_NAMES))
                )
            )

# create and save split texts
split_texts = {}
for i in range(len(ENTITIES)):
    split_texts[ENTITIES[i]] = {}
    for j in range(len(DYNAMICS)):
        split_texts[ENTITIES[i]][DYNAMICS[j]] = {}
        for k in range(len(ROLES)):
            split_texts[ENTITIES[i]][DYNAMICS[j]][ROLES[k]] = {}
            descriptors = all_texts[MESSENGER_ENTITIES[i]][ROLES[k]][
                MESSENGER_DYNAMICS[j]
            ]
            random.shuffle(descriptors)
            for l in range(len(SPLIT_NAMES)):
                start_idx = np.sum(proportional_counts[i, j, k, :l])
                end_idx = start_idx + proportional_counts[i, j, k, l]
                split_texts[ENTITIES[i]][DYNAMICS[j]][ROLES[k]][
                    SPLIT_NAMES[l]
                ] = descriptors[start_idx:end_idx]
with open(os.path.join(SAVE_DIR, "custom_text_splits.json"), "w") as f:
    json.dump(split_texts, f)

# verify that the texts are properly split
descriptors = []
for entity, entity_descriptors in split_texts.items():
    for dynamic, dynamic_descriptors in entity_descriptors.items():
        for role, role_descriptors in dynamic_descriptors.items():
            for split, split_descriptors in role_descriptors.items():
                for split_descriptor in split_descriptors:
                    assert split_descriptor not in descriptors
                    descriptors.append(split_descriptor)
print(len(descriptors))
print(descriptor_counts.sum())
assert len(descriptors) == descriptor_counts.sum()
print("verified")
