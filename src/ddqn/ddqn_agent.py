import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.environment.environment import create_env
from collections import deque, namedtuple
from typing import Tuple


# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, input_channels: int, action_size: int):
        super(QNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (64, 7, 7)
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Define a named tuple for storing experiences in the replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


# Define the Replay Memory class
class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define the DDQN Agent
class DQNAgent:
    def __init__(
            self,
            state_shape: Tuple[int],
            action_size: int,
            device: torch.device,
            gamma: float = 0.99,
            batch_size: int = 32,
            lr: float = 1e-4,
            replay_memory_size: int = 100000,
            target_update_frequency: int = 1000,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.1,
            epsilon_decay: int = 1000000,
            save_path: str = 'dqn_model.pth'  # Path to save the model
    ):
        self.device = device
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize networks
        self.policy_net = QNetwork(state_shape[0], action_size).to(device)
        self.target_net = QNetwork(state_shape[0], action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayMemory(replay_memory_size)

        # Epsilon parameters for epsilon-greedy policy
        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Target network update frequency
        self.target_update_frequency = target_update_frequency

        # Model saving parameters
        self.save_path = save_path

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < eps_threshold:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()

    def optimize_model(self):
        """Performs a single optimization step."""
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert batch elements to tensors
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

        # Compute Q(s_t, a_t)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Q targets for next states using Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)

        # Compute loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        """Updates the target network by copying weights from the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """Saves the policy network to a file."""
        torch.save(self.policy_net.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self, path: str):
        """Loads the policy network from a file."""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")


# Training loop
def main():
    num_episodes = 10000
    save_every = 100  # Save the model every 100 episodes
    env = create_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_size = env.action_space.n
    state_shape = env.observation_space.shape  # Should be (channels, height, width)

    agent = DQNAgent(state_shape, action_size, device)

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Convert states to tensors
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            agent.memory.push(state_tensor, action, reward, next_state_tensor, done)
            state = next_state
            agent.optimize_model()

            if agent.steps_done % agent.target_update_frequency == 0:
                agent.update_target_network()

        # Save the model at specified intervals
        if episode % save_every == 0:
            agent.save_model()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
