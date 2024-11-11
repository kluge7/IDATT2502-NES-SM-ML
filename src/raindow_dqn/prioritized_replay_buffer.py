import torch
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with N-step returns."""

    def __init__(self, capacity, alpha, n_step, gamma):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        self.priorities = deque(maxlen=capacity)

        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        """Stores a transition with N-step returns."""
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return
        # Compute N-step return
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]

        max_prio = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)

    def _get_n_step_info(self):
        """Calculates N-step return and next state."""
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def sample(self, batch_size, beta):
        """Samples a batch with importance-sampling weights."""
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        batch = list(zip(*samples))

        states = torch.cat(batch[0])
        actions = torch.tensor(batch[1])
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.cat(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        """Updates priorities based on TD error."""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio.detach().cpu().numpy()