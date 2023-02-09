import pickle
import numpy as np

class DataLoader:
    def __init__(self, dataset_path, data_name, batch_size, max_rollout_length):
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)[data_name]
        self.n_rollouts = len(self.data['manuals'])
        # keep only up to max_rollout_length
        for i in range(self.n_rollouts):
            self.data['grid_sequences'][i] = self.data['grid_sequences'][i][:max_rollout_length]
            self.data['action_sequences'][i] = self.data['action_sequences'][i][:max_rollout_length]
        self.rollout_lengths = np.array([len(sequence) for sequence in self.data['grid_sequences']])
        self.batch_size = batch_size

        # using these for quicker indexing
        self.manuals_array = np.zeros((self.n_rollouts), dtype=object)
        self.ground_truths_array = np.zeros((self.n_rollouts), dtype=object)
        self.grid_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths), 10, 10, 4), dtype=int)
        self.action_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths)), dtype=int)
        print("n_rollouts:", self.n_rollouts, "; max_rollout_length:", np.max(self.rollout_lengths))
        for i in range(self.n_rollouts):
            self.manuals_array[i] = self.data['manuals'][i]
            self.ground_truths_array[i] = self.data['ground_truths'][i]
            self.grid_sequences_array[i, :self.rollout_lengths[i]] = self.data['grid_sequences'][i]
            self.action_sequences_array[i, :self.rollout_lengths[i]] = self.data['action_sequences'][i]
    
    def reset(self):
        # randomly sample batch_size rollouts from data and keep track of their lengths
        self.indices = np.random.randint(low=0, high=self.n_rollouts, size=self.batch_size)
        self.remaining_lengths = self.rollout_lengths[self.indices] - 1
        return self.grid_sequences_array[self.indices, 0], self.action_sequences_array[self.indices, 0], self.manuals_array[self.indices], self.ground_truths_array[self.indices]

    def step(self):
        self.remaining_lengths -= 1
        new_idxs = np.argwhere(self.remaining_lengths < 0).squeeze(-1)
        cur_idxs = np.argwhere(self.remaining_lengths >= 0).squeeze(-1)
        self.indices[new_idxs] = np.random.randint(low=0, high=self.n_rollouts, size=len(new_idxs))
        self.remaining_lengths[new_idxs] = self.rollout_lengths[self.indices[new_idxs]] - 1
        return self.grid_sequences_array[self.indices, self.rollout_lengths[self.indices] - self.remaining_lengths - 1], self.action_sequences_array[self.indices, self.rollout_lengths[self.indices] - self.remaining_lengths - 1], self.manuals_array[self.indices], self.ground_truths_array[self.indices], (new_idxs, cur_idxs)
