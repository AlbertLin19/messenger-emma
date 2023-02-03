import pickle
import numpy as np

class DataLoader:
    def __init__(self, dataset_path, data_name, batch_size):
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)[data_name]
        self.n_rollouts = len(self.data['manuals'])
        self.rollout_lengths = np.array([len(sequence) for sequence in self.data['grid_sequences']])
        self.batch_size = batch_size

        # using these for quicker indexing
        self.manuals_array = np.zeros((self.n_rollouts), dtype=str)
        self.ground_truths_array = np.zeros((self.n_rollouts), dtype=str)
        self.grid_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths)), dtype=int)
        for i in range(self.n_rollouts):
            self.manuals_array[i] = self.data['manuals'][i]
            self.ground_truths_array[i] = self.data['ground_truths'][i]
            self.grid_sequences_array[i, :self.rollout_lengths[i]] = self.data['grid_sequences'][i]
    
    def reset(self):
        # randomly sample batch_size rollouts from data and keep track of their lengths
        self.indices = np.random.randint(low=0, high=self.n_rollouts, size=self.batch_size)
        self.remaining_lengths = self.rollout_lengths[self.indices] - 1
        return self.grid_sequences_array[self.indices, 0], self.manuals_array[self.indices], self.ground_truths_array[self.indices]

    def step(self):
        self.remaining_lengths -= 1
        new = self.remaining_lengths < 0
        self.indices[new] = np.random.randint(low=0, high=self.n_rollouts, size=np.sum(new))
        self.remaining_lengths[new] = self.rollout_lengths[self.indices[new]] - 1
        return self.grid_sequences_array[self.indices, self.rollout_lengths[self.indices] - self.remaining_lengths - 1], new
