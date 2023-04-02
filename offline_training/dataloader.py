import numpy as np
from tqdm import tqdm

CUSTOM_TO_MESSENGER_ENTITY = {
    "robot": "robot",
    "airplane": "airplane",
    "thief": "thief",
    "scientist": "scientist",
    "queen": "queen",
    "ship": "ship",
    "dog": "dog",
    "bird": "bird",
    "fish": "fish",
    "mage": "mage",
    "orb": "ball",
    "sword": "sword",
}
CUSTOM_TO_MESSENGER_DYNAMIC = {
    "chasing": "chaser",
    "fleeing": "fleeing",
    "immobile": "immovable",
}
CUSTOM_TO_MESSENGER_ROLE = {
    "message": "message",
    "enemy": "enemy",
    "goal": "goal",
}

def get_manual(dataset, split, i):
    texts = dataset["texts"]
    entities = dataset["keys"]["entities"]
    dynamics = dataset["keys"]["dynamics"]
    roles = dataset["keys"]["roles"]
    manual_idx = dataset["rollouts"][split]["manual_idxs"][i]
    ground_truth_idx = dataset["rollouts"][split]["ground_truth_idxs"][i]
    return [texts[entities[ground_truth_idx[j][0]]][dynamics[ground_truth_idx[j][1]]][roles[ground_truth_idx[j][2]]][split][manual_idx[j]] for j in range(len(manual_idx))]

def get_ground_truth(dataset, split, i):
    entities = dataset["keys"]["entities"]
    dynamics = dataset["keys"]["dynamics"]
    roles = dataset["keys"]["roles"]
    ground_truth_idx = dataset["rollouts"][split]["ground_truth_idxs"][i]
    return [(CUSTOM_TO_MESSENGER_ENTITY[entities[ground_truth_idx[j][0]]], CUSTOM_TO_MESSENGER_DYNAMIC[dynamics[ground_truth_idx[j][1]]], CUSTOM_TO_MESSENGER_ROLE[roles[ground_truth_idx[j][2]]]) for j in range(len(ground_truth_idx))]

# takes in custom dataset (which has its own custom keywords) 
# and outputs in terms of Messenger keywords
class DataLoader:
    def __init__(self, dataset, split, max_rollout_length):
        self.n_rollouts = len(dataset["rollouts"][split]["manual_idxs"])
        self.rollout_lengths = np.array([min(len(sequence), max_rollout_length) for sequence in dataset["rollouts"][split]["grid_sequences"]])
        self.rollout_probs = (self.rollout_lengths-1)/np.sum(self.rollout_lengths-1)

        self.manuals_array = np.zeros((self.n_rollouts), dtype=object)
        self.ground_truths_array = np.zeros((self.n_rollouts), dtype=object)
        self.action_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths)), dtype=int)
        self.reward_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths)), dtype=float)
        self.grid_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths), 10, 10, 4), dtype=int)
        for i in tqdm(range(self.n_rollouts)):
            self.manuals_array[i] = get_manual(dataset, split, i)
            self.ground_truths_array[i] = get_ground_truth(dataset, split, i)
            self.action_sequences_array[i, :self.rollout_lengths[i]] = dataset["rollouts"][split]["action_sequences"][i][:max_rollout_length]
            self.reward_sequences_array[i, :self.rollout_lengths[i]] = dataset["rollouts"][split]["reward_sequences"][i][:max_rollout_length]
            self.grid_sequences_array[i, :self.rollout_lengths[i]] = dataset["rollouts"][split]["grid_sequences"][i][:max_rollout_length]
        print("n_rollouts:", self.n_rollouts, "; max_rollout_length:", np.max(self.rollout_lengths))
    
    def reset(self, mode, batch_size):
        self.mode = mode
        if self.mode == "random":
            # randomly sample batch_size rollouts from data (with probability proportional to their lengths) and keep track of their lengths
            self.indices = np.random.choice(self.n_rollouts, size=batch_size, p=self.rollout_probs)
            # randomly sample starting state for each rollout
            self.timesteps = (np.random.rand(batch_size)*(self.rollout_lengths[self.indices] - 1)).astype(int)
        else:
            raise NotImplementedError
        return self.manuals_array[self.indices], self.ground_truths_array[self.indices], self.grid_sequences_array[self.indices, self.timesteps]

    def step(self):
        self.timesteps += 1
        new_idxs = np.argwhere(self.timesteps >= self.rollout_lengths[self.indices]).squeeze(-1)
        cur_idxs = np.argwhere(self.timesteps < self.rollout_lengths[self.indices]).squeeze(-1)
        if self.mode == "random":
            self.indices[new_idxs] = np.random.choice(self.n_rollouts, size=len(new_idxs), p=self.rollout_probs)
            self.timesteps[new_idxs] = (np.random.rand(len(new_idxs))*(self.rollout_lengths[self.indices[new_idxs]] - 1)).astype(int)
        else:
            raise NotImplementedError

        return (
            self.manuals_array[self.indices], 
            self.ground_truths_array[self.indices], 
            self.action_sequences_array[self.indices, self.timesteps], 
            self.grid_sequences_array[self.indices, self.timesteps], 
            self.reward_sequences_array[self.indices, self.timesteps], 
            (new_idxs, cur_idxs), 
            self.timesteps
        )