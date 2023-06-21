import os
import json
import math
import matplotlib.pyplot as plt

MODEL_NAMES = {
    'embed': 'EMMA',
    'gpt': "ChatGPT",
    "oracle": "Oracle",
    "none": "Observational"
}

MODEL_COLORS = {
    "embed": 'blue',
    'gpt': 'green',
    'oracle': 'gold',
    'none': 'red'
}

output_folder = 'plots'

folder = 'evaluation'
subfolders = ['real', 'imagined']

for subfolder in subfolders:
    for split in os.listdir(os.path.join(folder, subfolder)):
        fig, ax = plt.subplots(figsize=(4, 3))

        # for model in os.listdir(os.path.join(folder, subfolder, split)):
        for model in ['none', 'oracle', 'gpt', 'embed']:
            with open(os.path.join(folder, subfolder, split, model, split.replace('dev', 'test').replace('best_loc_loss', 'loc_ces')+'.json'), 'r') as f:
                ces = json.load(f)
                ax.plot(range(1, len(ces)+1), ces, c=MODEL_COLORS[model], label=MODEL_NAMES[model])
        
        ax.set_title('Messenger')
        ax.legend(title='Model')
        ax.set_ylabel('Average Cross Entropy')
        ax.set_xlabel('Number of Steps Rolled Out')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, subfolder, split+'.jpg'), dpi=800)

        # plot perplexity instead of cross entropy
        fig, ax = plt.subplots(figsize=(4, 3))

        # for model in os.listdir(os.path.join(folder, subfolder, split)):
        for model in ['none', 'oracle', 'gpt', 'embed']:
            with open(os.path.join(folder, subfolder, split, model, split.replace('dev', 'test').replace('best_loc_loss', 'loc_ces')+'.json'), 'r') as f:
                ces = json.load(f)
                ax.plot(range(1, len(ces)+1), [math.exp(ce) for ce in ces], c=MODEL_COLORS[model], label=MODEL_NAMES[model])
        
        ax.set_title('Messenger')
        ax.legend(title='Model')
        ax.set_ylabel('Average Perplexity')
        ax.set_xlabel('Number of steps rolled out')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, subfolder, split+'_perplexity.jpg'), dpi=800)