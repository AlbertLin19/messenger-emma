import os
import json
import random
import logging
import pickle
from collections import defaultdict

import numpy as np
import torch

from chatgpt_groundings.utils import ENTITY_GROUNDING_LOOKUP, MOVEMENT_GROUNDING_LOOKUP, ROLE_GROUNDING_LOOKUP

# mappings from Jens keywords to Messenger keywords
CUSTOM_TO_MESSENGER_ENTITY = {
    'robot': 'robot',
    'airplane': 'airplane',
    'thief': 'thief',
    'scientist': 'scientist',
    'queen': 'queen',
    'ship': 'ship',
    'dog': 'dog',
    'bird': 'bird',
    'fish': 'fish',
    'mage': 'mage',
    'orb': 'ball',
    'sword': 'sword',
}
CUSTOM_TO_MESSENGER_DYNAMIC = {
    'chasing': 'chaser',
    'fleeing': 'fleeing',
    'immobile': 'immovable',
}
CUSTOM_TO_MESSENGER_ROLE = {
    'message': 'message',
    'enemy': 'enemy',
    'goal': 'goal',
}

# helper function to retrieve the manual of ith game in dataset's split
def get_manual(dataset, split, i):
    texts = dataset['texts']
    entities = dataset['keys']['entities']
    dynamics = dataset['keys']['dynamics']
    roles = dataset['keys']['roles']
    manual_idx = dataset['rollouts'][split]['manual_idxs'][i]
    ground_truth_idx = dataset['rollouts'][split]['ground_truth_idxs'][i]
    return [texts[entities[ground_truth_idx[j][0]]][dynamics[ground_truth_idx[j][1]]][roles[ground_truth_idx[j][2]]][split][manual_idx[j]] for j in range(len(manual_idx))]

def get_gpt_parsed_manual(gpt_groundings, manual):
    return [gpt_groundings[s] for s in manual]

# helper function to retrieve the ground truth of ith game in dataset's split
def get_true_parsed_manual(dataset, split, i):
    entities = dataset['keys']['entities']
    dynamics = dataset['keys']['dynamics']
    roles = dataset['keys']['roles']
    ground_truth_idx = dataset['rollouts'][split]['ground_truth_idxs'][i]
    return [(CUSTOM_TO_MESSENGER_ENTITY[entities[ground_truth_idx[j][0]]], CUSTOM_TO_MESSENGER_DYNAMIC[dynamics[ground_truth_idx[j][1]]], CUSTOM_TO_MESSENGER_ROLE[roles[ground_truth_idx[j][2]]]) for j in range(len(ground_truth_idx))]


class Dataset:

    def __init__(self, config, seed=None):

        with open(config.dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        with open(config.gpt_groundings_path, 'r') as f:
            gpt_groundings = json.load(f)
            for s, grounding in gpt_groundings.items():
                gpt_groundings[s] = (
                    ENTITY_GROUNDING_LOOKUP[grounding[0]],
                    MOVEMENT_GROUNDING_LOOKUP[grounding[1]],
                    ROLE_GROUNDING_LOOKUP[grounding[2]]
                )

        self.data = {}
        for split in dataset['rollouts']:
            data = []
            split_size = len(dataset['rollouts'][split]['manual_idxs'])
            for i in range(split_size):
                item = {}
                item['id'] = '%s_%d' % (split, i)
                for k in dataset['rollouts'][split]:
                    name = '_'.join(k.split('_')[:-1])
                    if name == 'manual':
                        item['manual'] = get_manual(dataset, split, i)
                        item['gpt_parsed_manual'] = get_gpt_parsed_manual(gpt_groundings, item['manual'])
                    elif name == 'ground_truth':
                        item['true_parsed_manual'] = get_true_parsed_manual(dataset, split, i)
                    else:
                        item[name] = np.array(dataset['rollouts'][split][k][i][:config.max_rollout_length])
                data.append(item)

            self.data[split] = DataSplit(data, config.device, seed=seed)
            if 'train' in split:
                self.data[split.replace('train', 'train_dev')] = DataSplit(
                    self.data[split].data[:100], config.device, seed=seed)

        for split in self.data:
            print('%s split contains %d examples' % (split, len(self.data[split])))

        self.splits = list(self.data.keys())

    def __getitem__(self, split):
        return self.data[split]

    def __iter__(self):
        return iter(self.splits)

    def __next__(self):
        return next(self.split_iter)


class DataSplit:

    def __init__(self, data, device, seed=None):

        self.data = data
        self.device = device
        self.random = random.Random(seed)
        self.random.shuffle(data)

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        self.random.shuffle(self.data)

    def random_item(self):
        return self.random.choice(self.data)

    def random_batch(self, batch_size):
        return self.prepare_batch(self.random.sample(self.data, batch_size))

    def iterate_batches(self, batch_size=1, cycle=False):

        self.idx = 0

        while True:

            batch = self.data[self.idx : (self.idx + batch_size)]

            if len(batch) < batch_size:
                batch += self.random.sample(self.data, batch_size - len(batch))

            self.idx += batch_size
            if self.idx >= len(self.data):
                self.random.shuffle(self.data)
                self.idx = 0

            if batch:
                yield self.prepare_batch(batch)

            if not cycle and self.idx == 0:
                break

    def prepare_batch(self, batch):

        batch_size = len(batch)

        new_batch = defaultdict(list)

        lens = []
        for item in batch:
            lens.append(item['grid'].shape[0])

        max_len = max(lens)

        mask = torch.ones((max_len + 1, batch_size)).to(self.device).bool()
        for i in range(batch_size):
            mask[:lens[i], i] = 0

        for item in batch:
            for k in item:
                if k == 'id' or 'manual' in k:
                    new_batch[k].append(item[k])
                else:
                    x = torch.from_numpy(item[k]).to(self.device)
                    if k == 'reward':
                        x = x.float()
                    else:
                        x = x.long()
                    pad = x[-1:].expand(max_len + 1 - x.shape[0], *x.shape[1:])
                    # set last action to 0 (stay)
                    if k == 'action':
                        pad = torch.zeros_like(pad)
                    x = torch.cat([x, pad], dim=0)
                    new_batch[k].append(x)

        for k in new_batch:
            if k != 'id' and 'manual' not in k:
                new_batch[k] = torch.stack(new_batch[k])
                #new_batch[k] = new_batch[k].transpose(0, 1)
        new_batch['mask'] = mask
        new_batch['len'] = lens

        return new_batch


