import os
import json
import matplotlib.pyplot as plt

MODEL_NAMES = {
    "embed": "EMMA",
    "gpt": "ChatGPT",
    "oracle": "Oracle",
    "none": "Observational",
}

MODEL_COLORS = {"embed": "blue", "gpt": "green", "oracle": "gold", "none": "red"}

output_folder = "plots"

folder = "evaluation"
subfolders = ["real", "imagined"]

for subfolder in subfolders:
    for split in os.listdir(os.path.join(folder, subfolder)):
        fig, ax = plt.subplots(figsize=(4, 3))

        # for model in os.listdir(os.path.join(folder, subfolder, split)):
        for model in ["none", "oracle", "gpt", "embed"]:
            with open(
                os.path.join(
                    folder,
                    subfolder,
                    split,
                    model,
                    split.replace("dev", "test").replace("best_loc_loss", "loc_perps")
                    + ".json",
                ),
                "r",
            ) as f:
                perps = json.load(f)
                ax.plot(
                    range(1, len(perps) + 1),
                    perps,
                    c=MODEL_COLORS[model],
                    label=MODEL_NAMES[model],
                )

        ax.set_title("Messenger")
        ax.legend(title="Model")
        ax.set_ylabel("Average Perplexity")
        ax.set_xlabel("Number of Steps Rolled Out")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, subfolder, split + ".jpg"), dpi=800)
