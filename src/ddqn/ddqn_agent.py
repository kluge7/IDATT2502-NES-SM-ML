import math
import random

import hyperparameters as hp
import torch
import torch.nn.functional as F
import torch.optim as optim
from q_network import QNetwork
from replay_memory import ReplayMemory, Transition


class DQNAgent:
    def __init__(self, state_shape, action_size, device):
        self.device = device
        self.action_size = action_size
        self.gamma = hp.GAMMA
        self.batch_size = hp.BATCH_SIZE
        self.target_update_frequency = hp.TARGET_UPDATE_FREQUENCY

        self.policy_net = QNetwork(state_shape[0], action_size).to(device)
        self.target_net = QNetwork(state_shape[0], action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hp.LR)
        self.memory = ReplayMemory(hp.REPLAY_MEMORY_SIZE)

        self.steps_done = 0
        self.epsilon_start = hp.EPSILON_START
        self.epsilon_end = hp.EPSILON_END
        self.epsilon_decay = hp.EPSILON_DECAY

    def get_epsilon_threshold(self):
        """Calculates the current epsilon threshold for epsilon-greedy action selection.

        This threshold decreases over time to reduce exploration in favor of exploitation as training progresses.

        Returns:
            float: The current epsilon threshold for selecting random versus greedy actions.
        """
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

    def select_action(self, state):
        """Selects an action based on the epsilon-greedy strategy and the current policy network.

        Args:
            state (array-like): The current state of the environment.

        Returns:
            int: The chosen action, either random (exploration) or the action with the highest Q-value (exploitation).
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
        """Performs a single training step by optimizing the policy network with a sampled batch of transitions.

        The update uses the Bellman equation to adjust the Q-values in the policy network toward expected values.
        """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_state_values = (
                self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            )
            expected_state_action_values = (
                reward_batch + self.gamma * next_state_values * (1 - done_batch)
            )

        loss = F.smooth_l1_loss(
            state_action_values.squeeze(1), expected_state_action_values
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        """Copies the weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """Saves the policy network's state dictionary to a specified file path."""
        torch.save(self.policy_net.state_dict(), hp.MODEL_SAVE_PATH)
        print(f"Model saved to {hp.MODEL_SAVE_PATH}")

    def load_model(self, path):
        """Loads the policy network's state dictionary from a specified file path and sets up the target network.

        Args:
            path (str): The file path from which to load the model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path} on {device}")
