import numpy as np
from tqdm import tqdm

# mappings from Jens keywords to Messenger keywords
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

# helper function to retrieve the manual of ith game in dataset's split
def get_manual(dataset, split, i):
    texts = dataset["texts"]
    entities = dataset["keys"]["entities"]
    dynamics = dataset["keys"]["dynamics"]
    roles = dataset["keys"]["roles"]
    manual_idx = dataset["rollouts"][split]["manual_idxs"][i]
    ground_truth_idx = dataset["rollouts"][split]["ground_truth_idxs"][i]
    return [texts[entities[ground_truth_idx[j][0]]][dynamics[ground_truth_idx[j][1]]][roles[ground_truth_idx[j][2]]][split][manual_idx[j]] for j in range(len(manual_idx))]

# helper function to retrieve the ground truth of ith game in dataset's split
def get_ground_truth(dataset, split, i):
    entities = dataset["keys"]["entities"]
    dynamics = dataset["keys"]["dynamics"]
    roles = dataset["keys"]["roles"]
    ground_truth_idx = dataset["rollouts"][split]["ground_truth_idxs"][i]
    return [(CUSTOM_TO_MESSENGER_ENTITY[entities[ground_truth_idx[j][0]]], CUSTOM_TO_MESSENGER_DYNAMIC[dynamics[ground_truth_idx[j][1]]], CUSTOM_TO_MESSENGER_ROLE[roles[ground_truth_idx[j][2]]]) for j in range(len(ground_truth_idx))]

# takes in custom dataset (which has its own custom keywords)
# and outputs batched data step-by-step (using Messenger keywords)
class DataLoader:
    def __init__(self, dataset, split, max_rollout_length, mode, start_state, batch_size, max_rollouts=1e8):
        self.n_rollouts = min(len(dataset["rollouts"][split]["manual_idxs"]), max_rollouts)
        self.rollout_lengths = np.array(
            [
                min(len(sequence), max_rollout_length)
                    for sequence in dataset["rollouts"][split]["grid_sequences"][:self.n_rollouts]
            ]
        )
        self.rollout_probs = (self.rollout_lengths-1)/np.sum(self.rollout_lengths-1)

        #print(dataset['rollouts']['train_games']['grid_sequences'][41][0][:,:,3])
        #print(dataset['rollouts']['train_games']['grid_sequences'][41][1][:,:,3])

        # cache data
        self.manuals_array = np.zeros((self.n_rollouts), dtype=object)
        self.ground_truths_array = np.zeros((self.n_rollouts), dtype=object)
        self.action_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths)), dtype=int)
        self.reward_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths)), dtype=float)
        self.done_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths)), dtype=bool)
        self.grid_sequences_array = np.zeros((self.n_rollouts, np.max(self.rollout_lengths), 10, 10, 4), dtype=int)
        for i in tqdm(range(self.n_rollouts)):
            self.manuals_array[i] = get_manual(dataset, split, i)
            self.ground_truths_array[i] = get_ground_truth(dataset, split, i)
            self.action_sequences_array[i, :self.rollout_lengths[i]] = dataset["rollouts"][split]["action_sequences"][i][:max_rollout_length]
            self.reward_sequences_array[i, :self.rollout_lengths[i]] = dataset["rollouts"][split]["reward_sequences"][i][:max_rollout_length]
            self.done_sequences_array[i, :self.rollout_lengths[i]] = dataset["rollouts"][split]["done_sequences"][i][:max_rollout_length]
            self.grid_sequences_array[i, :self.rollout_lengths[i]] = dataset["rollouts"][split]["grid_sequences"][i][:max_rollout_length]

        print(split, "n_rollouts:", self.n_rollouts, "; max_rollout_length:", np.max(self.rollout_lengths))

        # sampling method for rollouts
        self.mode = mode
        assert mode in ["random", "static"]

        # sampling method for initial states
        self.start_state = start_state
        assert start_state in ["initial", "anywhere"]

        self.batch_size = batch_size

    # retrieve initial-state data
    def reset(self):
        if self.mode == "random":
            if self.start_state == "initial":
                # randomly sample batch_size rollouts from data (with equal probability) and keep track of their lengths
                self.indices = np.random.choice(self.n_rollouts, size=self.batch_size)
                # start at initial state for each rollout
                self.timesteps = np.zeros((self.batch_size), dtype=int)
            elif self.start_state == "anywhere":
                # randomly sample batch_size rollouts from data (with probability proportional to their lengths) and keep track of their lengths
                self.indices = np.random.choice(self.n_rollouts, size=self.batch_size, p=self.rollout_probs)
                # randomly sample starting state for each rollout
                self.timesteps = (np.random.rand(self.batch_size)*(self.rollout_lengths[self.indices] - 1)).astype(int)
            else:
                raise NotImplementedError
            return self.manuals_array[self.indices], self.ground_truths_array[self.indices], self.grid_sequences_array[self.indices, self.timesteps]
        elif self.mode == "static":
            # keep track of available rollout indices
            self.avail_indices = np.arange(self.n_rollouts, dtype=int)
            # take initial set of rollouts
            self.indices = -1*np.ones((self.batch_size), dtype=int)
            if self.batch_size < len(self.avail_indices):
                self.indices[:self.batch_size] = self.avail_indices[:self.batch_size]
                self.avail_indices = self.avail_indices[self.batch_size:]
            else:
                self.indices[:len(self.avail_indices)] = self.avail_indices
                self.avail_indices = None
            if self.start_state == "initial":
                # start at initial state for each rollout
                self.timesteps = np.zeros((self.batch_size), dtype=int)
            else:
                raise NotImplementedError
            nonnegative_indices = self.indices + (self.indices < 0)
            return self.manuals_array[nonnegative_indices], self.ground_truths_array[nonnegative_indices], self.grid_sequences_array[nonnegative_indices, self.timesteps], self.n_rollouts
        else:
            raise NotImplementedError

    # retrieve next-step data
    def step(self):
        if self.mode == "random":
            self.timesteps += 1
            #print(self.timesteps)
            #print(self.rollout_lengths[self.indices])
            new_idxs = np.argwhere(self.timesteps >= self.rollout_lengths[self.indices]).squeeze(-1)
            cur_idxs = np.argwhere(self.timesteps < self.rollout_lengths[self.indices]).squeeze(-1)
            if self.start_state == "initial":
                # randomly sample batch_size rollouts from data (with equal probability) and keep track of their lengths
                self.indices[new_idxs] = np.random.choice(self.n_rollouts, size=len(new_idxs))
                # start at initial state for each rollout
                self.timesteps[new_idxs] = 0
            elif self.start_state == "anywhere":
                # randomly sample batch_size rollouts from data (with probability proportional to their lengths) and keep track of their lengths
                self.indices[new_idxs] = np.random.choice(self.n_rollouts, size=len(new_idxs), p=self.rollout_probs)
                self.timesteps[new_idxs] = (np.random.rand(len(new_idxs))*(self.rollout_lengths[self.indices[new_idxs]] - 1)).astype(int)
            else:
                raise NotImplementedError

            #print(self.indices[0], self.timesteps[0])
            #print(self.grid_sequences_array[self.indices[0], self.timesteps[0]][:,:,3])

            return (
                self.manuals_array[self.indices],
                self.ground_truths_array[self.indices],
                self.action_sequences_array[self.indices, self.timesteps],
                self.grid_sequences_array[self.indices, self.timesteps],
                self.reward_sequences_array[self.indices, self.timesteps],
                self.done_sequences_array[self.indices, self.timesteps],
                (new_idxs, cur_idxs),
                self.timesteps,
            )
        elif self.mode == "static":
            self.timesteps += (self.indices >= 0)
            nonnegative_indices = self.indices + (self.indices < 0)
            new_idxs = np.argwhere((self.indices >= 0)*(self.timesteps >= self.rollout_lengths[nonnegative_indices])).squeeze(-1)
            cur_idxs = np.argwhere((self.indices >= 0)*(self.timesteps < self.rollout_lengths[nonnegative_indices])).squeeze(-1)
            self.indices[new_idxs] = -1
            if self.start_state == "initial":
                # start at initial state for each rollout
                self.timesteps[new_idxs] = 0
            else:
                raise NotImplementedError
            # retrieve unused rollouts if any still remain
            if self.avail_indices is not None:
                if len(new_idxs) < len(self.avail_indices):
                    self.indices[new_idxs] = self.avail_indices[:len(new_idxs)]
                    self.avail_indices = self.avail_indices[len(new_idxs):]
                else:
                    new_idxs = new_idxs[:len(self.avail_indices)]
                    self.indices[new_idxs] = self.avail_indices
                    self.avail_indices = None
            nonnegative_indices = self.indices + (self.indices < 0)
            return (
                self.manuals_array[nonnegative_indices],
                self.ground_truths_array[nonnegative_indices],
                self.action_sequences_array[nonnegative_indices, self.timesteps],
                self.grid_sequences_array[nonnegative_indices, self.timesteps],
                self.reward_sequences_array[nonnegative_indices, self.timesteps],
                self.done_sequences_array[nonnegative_indices, self.timesteps],
                (new_idxs, cur_idxs),
                self.timesteps,
                (self.indices >= 0)*(self.timesteps == (self.rollout_lengths[nonnegative_indices]-1)), # whether rollout was just completed
                (self.indices < 0).all(), # whether all rollouts have been completed
            )
        else:
            raise NotImplementedError
