import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.ddqn.hyperparameters import DDQNHyperparameters
from src.ddqn.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.network.cnn_network import CNNNetwork


class DDQNAgent:
    def __init__(self, state_shape, action_size, device, hp: DDQNHyperparameters):
        self.device = device
        self.action_size = action_size
        self.gamma = hp.gamma
        self.batch_size = hp.batch_size
        self.target_update_frequency = hp.target_update_frequency

        self.policy_net = CNNNetwork(state_shape[0], action_size).to(device)
        self.target_net = CNNNetwork(state_shape[0], action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hp.lr)
        self.memory = PrioritizedReplayBuffer(hp.replay_memory_size, hp.priority_alpha)

        self.steps_done = 0
        self.epsilon_start = hp.epsilon_start
        self.epsilon_end = hp.epsilon_end
        self.epsilon_decay = hp.epsilon_decay
        self.beta = hp.priority_beta_start
        self.beta_frames = hp.priority_beta_frames
        self.model_save_path = hp.model_save_path
        self.csv_filename = hp.csv_filename

    def get_epsilon_threshold(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy.

        Args:
            state (numpy.ndarray): The current state.

        Returns:
            int: Selected action index.
        """
        eps_threshold = self.get_epsilon_threshold()
        self.steps_done += 1
        if random.random() < eps_threshold:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()

    def update(self):
        """Updates the policy network.

        This is done by sampling a batch from the replay buffer, computing
        the loss, and performing a gradient descent step. Uses Double DQN to calculate
        target values.
        """
        if (
            len(self.memory) < self.memory.capacity / 4
        ):  # Train only after buffer is 1/4 full
            return

        self.beta = min(1.0, self.beta + (1.0 - self.beta) / self.beta_frames)

        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.batch_size, self.beta)
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        state_action_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_state_values = (
                self.target_net(next_states).gather(1, next_actions).squeeze(1)
            )
            expected_state_action_values = rewards + self.gamma * next_state_values * (
                1 - dones
            )

        td_errors = state_action_values.squeeze(1) - expected_state_action_values
        loss = (
            weights
            * F.smooth_l1_loss(
                state_action_values.squeeze(1),
                expected_state_action_values,
                reduction="none",
            )
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        priorities = (
            td_errors.abs().detach().cpu().numpy() + 1e-5
        )  # Add small constant to avoid zero priorities
        self.memory.update_priorities(indices, priorities)

    def update_target_network(self):
        """Updates the target network by copying the weights from the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_save_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path} on {device}")
