import random
from collections import namedtuple

import numpy as np
import torch
from sum_tree import SumTree

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6  # Small constant to avoid zero priority

    def push(self, *args):
        """Saves a transition with maximum priority."""
        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = 1.0

        self.tree.add(max_priority, Transition(*args))

    def sample(self, batch_size, beta):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            priorities.append(priority)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()  # Normalize

        is_weight = torch.tensor(is_weight, dtype=torch.float32)
        return batch, idxs, is_weight

    def update_priorities(self, idxs, priorities):
        # Ensure priorities is a list or array, not a single float value
        if isinstance(priorities, (np.float64, float)):
            priorities = [priorities] * len(idxs)
        for idx, priority in zip(idxs, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries
