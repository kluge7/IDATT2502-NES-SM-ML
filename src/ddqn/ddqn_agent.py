import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import os
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


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


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
            epsilon_start: float = 0.1,
            epsilon_end: float = 0.1,
            epsilon_decay: int = 100000,
            save_path: str = 'dqn_model_new.pth'  # Path to save the model
    ):
        self.device = device
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_net = QNetwork(state_shape[0], action_size).to(device)
        self.target_net = QNetwork(state_shape[0], action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_memory_size)

        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.save_path = save_path

    def get_epsilon_threshold(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               math.exp(-1. * self.steps_done / self.epsilon_decay)

    def select_action(self, state):
        eps_threshold = self.get_epsilon_threshold()
        self.steps_done += 1
        if random.random() < eps_threshold:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()

    def optimize_model(self):
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
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)

        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self, path: str):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")


def main():
    # Define number of episodes, save frequency, etc.
    num_episodes = 10000
    save_every = 100
    csv_filename = 'training_log_new.csv'
    model_path = 'dqn_model_new.pth'

    # Environment and agent setup
    env = create_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_size = env.action_space.n
    state_shape = env.observation_space.shape
    agent = DQNAgent(state_shape, action_size, device)

    # Attempt to load model
    if os.path.exists(model_path):
        agent.load_model(model_path)

    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Episode', 'Total Reward', 'Flag Reached', 'Epsilon', 'Max Distance %']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for episode in range(1, num_episodes + 1):
            state = env.reset()
            total_reward = 0
            done = False
            flag_reached = False
            max_x_pos = 0
            goal_x_pos = 3260  # Hypothetical max x-position for flag

            while not done:
                eps_threshold = agent.get_epsilon_threshold()
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward

                max_x_pos = max(max_x_pos, info.get('x_pos', 0))
                if info.get('flag_get', False):
                    flag_reached = True

                state_tensor = torch.tensor(state, dtype=torch.float32)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

                agent.memory.push(state_tensor, action, reward, next_state_tensor, done)
                state = next_state
                agent.optimize_model()

                if agent.steps_done % agent.target_update_frequency == 0:
                    agent.update_target_network()

            max_distance_percentage = (max_x_pos / goal_x_pos) * 100

            if episode % save_every == 0:
                agent.save_model()

            writer.writerow({
                'Episode': episode,
                'Total Reward': total_reward,
                'Flag Reached': flag_reached,
                'Epsilon': eps_threshold,
                'Max Distance %': max_distance_percentage
            })
            csvfile.flush()

            print(f"Episode {episode}, Total Reward: {total_reward}, Flag Reached: {flag_reached}, "
                  f"Epsilon: {eps_threshold:.4f}, Max Distance %: {max_distance_percentage:.2f}")

    env.close()


if __name__ == "__main__":
    main()
