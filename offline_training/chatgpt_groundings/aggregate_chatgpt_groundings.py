import os
import json

SAVE_DIR = "../../messenger/envs/texts/chatgpt_groundings"
TEXT_FILES = ["text_train.json", "text_val.json", "text_test.json"]

version = 0
chatgpt_grounding_path = os.path.join(
    SAVE_DIR, "chatgpt_grounding_for_" + TEXT_FILES[0]
)
while os.path.isfile(chatgpt_grounding_path):
    print("version", version)
    chatgpt_groundings = {}
    for text_file in TEXT_FILES:
        with open(chatgpt_grounding_path[: -len(TEXT_FILES[0])] + text_file, "r") as f:
            chatgpt_groundings.update(json.load(f))
    with open(
        chatgpt_grounding_path[: -len(TEXT_FILES[0])] + "text_all.json", "w"
    ) as f:
        json.dump(chatgpt_groundings, f)
    version += 1
    chatgpt_grounding_path = os.path.join(
        SAVE_DIR, f"refined_{version}_chatgpt_grounding_for_" + TEXT_FILES[0]
    )
