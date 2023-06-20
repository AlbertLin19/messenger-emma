# Environment Setup
Implementation of the custom version of the Messenger environment used in the paper submission to NeurIPS 2023: [Language-Guided World Models](./#). Modified from the original Messenger environment and EMMA model from the ICML 2021 paper: [Grounding Language to Entities and Dynamics for Generalization in Reinforcement Learning](https://arxiv.org/abs/2101.07393). 

## Installation
Currently, only local installations are supported. Clone the repository and run:
```
pip install -e messenger-emma
```
This will install only the Messenger gym environments and dependencies. If you want to use models, you must install additional dependencies such as `torch`. Run the following instead:
```
pip install -e 'messenger-emma[models]'
```

## Usage
To instantiate a gym environment, use:
```python
import gym
import messenger
env = gym.make('msgr-custom-v2', shuffle_obs=False)
obs, manual, ground_truth = env.reset(split=<data split>, entities=<game entities>) 
obs, reward, done, info = env.step(<some action>)
```
Here `<data split>` should be a split name from the text splits file [custom_text_splits.json](./messenger/envs/texts/custom_text_splits/custom_text_splits.json), e.g ```'train_games'```. `<game entities>` should be a list of 3 items, where each item is itself a list specifying an entity name, movement type, and role type. Examples in the right formatting are found in the data splits file [data_splits_final_with_test.json](./offline_training/custom_dataset/data_splits_final_with_test.json). `<some action>` should be an integer between 0 and 4 corresponding to the actions `up`,`down`,`left`,`right`,`stay`. Notice that in contrast to standard gym, `env.reset()` returns a tuple of an observation, the text manual sampled for the current episode, and the ground truth information.

# World Model Training and Evaluation
Scripts for training and evaluating world models are found in the [offline_training](./offline_training/) folder.

# Policy + World Model Training and Evaluation
Scripts for training and evaluating policies with world models are found in the [policy_training](./policy_training/) folder.