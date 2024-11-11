import math
import random
import os
import csv  # Added for CSV functionality
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.environment.wrappers import (
    ConvertToTensor,
    CustomReward,
    FrameSkipper,
    Monitor,
    NormalizePixels,
    ObservationBuffer,
    ResizeAndGrayscale,
)

# Define your environment setup
def create_env(map="SuperMarioBros-v0", skip=4, output_path=None):
    """Sets up the Super Mario Bros environment with customized wrappers."""
    env = JoypadSpace(gym_super_mario_bros.make(map), COMPLEX_MOVEMENT)
    if output_path is not None:
        monitor = Monitor(width=256, height=240, saved_path=output_path)
    else:
        monitor = None
    env = CustomReward(env, monitor=monitor)
    env = FrameSkipper(env, skip=skip)
    env = ResizeAndGrayscale(env)
    env = ConvertToTensor(env)
    env = ObservationBuffer(env, 4)
    env = NormalizePixels(env)
    return env

# Implement NoisyLinear layer for Noisy Nets
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)

# Define the Rainbow DQN network
class RainbowDQN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int, num_atoms: int = 51, Vmin: float = -10, Vmax: float = 10):
        super(RainbowDQN, self).__init__()

        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (64, 7, 7)
            nn.ReLU(),
        )

        self.fc_input_dim = 64 * 7 * 7

        # Noisy layers
        self.noisy_value1 = NoisyLinear(self.fc_input_dim, 512)
        self.noisy_value2 = NoisyLinear(512, self.num_atoms)

        self.noisy_advantage1 = NoisyLinear(self.fc_input_dim, 512)
        self.noisy_advantage2 = NoisyLinear(512, self.num_actions * self.num_atoms)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        q_atoms = F.softmax(q_atoms, dim=2)
        q_atoms = q_atoms.clamp(min=1e-3)  # For numerical stability

        return q_atoms

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

# Implement Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.stack(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.stack(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio ** self.alpha

# Define the Agent class
class Agent:
    def __init__(self, env, input_channels, num_actions, device, Vmin=-10, Vmax=10, num_atoms=51, n_steps=3, gamma=0.99,
                 batch_size=32, replay_capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000, learning_rate=1e-4,
                 target_update=1000, save_path='checkpoints'):
        self.env = env
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.alpha = alpha
        self.learn_step_counter = 0
        self.target_update = target_update

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.policy_net = RainbowDQN(input_channels, num_actions, num_atoms=num_atoms, Vmin=Vmin, Vmax=Vmax).to(device)
        self.target_net = RainbowDQN(input_channels, num_actions, num_atoms=num_atoms, Vmin=Vmin, Vmax=Vmax).to(device)
        self.load_model()  # Load the latest model if available
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.replay_buffer = PrioritizedReplayBuffer(replay_capacity, alpha)
        self.n_step_buffer = []

        self.Vmin = Vmin
        self.Vmax = Vmax
        self.num_atoms = num_atoms
        self.delta_z = (Vmax - Vmin) / (num_atoms - 1)
        self.support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms).to(device)

        self.episode_counter = 0  # Initialize episode counter

        # Initialize CSV file
        self.csv_file = os.path.join(self.save_path, 'training_log.csv')
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Reward', 'Got Flag'])

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_atoms = self.policy_net(state)
            q_values = (q_atoms * self.support).sum(dim=2)
            action = q_values.argmax(1).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_steps:
            return
        R = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_steps)])
        state_n, action_n = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        next_state_n = self.n_step_buffer[-1][3]
        done_n = self.n_step_buffer[-1][4]

        self.replay_buffer.push(state_n, action_n, R, next_state_n, done_n)
        self.n_step_buffer.pop(0)

    def update_beta(self, frame_idx):
        beta = self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames
        return min(1.0, beta)

    def learn(self, frame_idx):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        beta = self.update_beta(frame_idx)

        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, beta)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        with torch.no_grad():
            next_q_atoms = self.policy_net(next_states)
            next_q_values = (next_q_atoms * self.support).sum(dim=2)
            next_actions = next_q_values.argmax(dim=1)

            next_q_atoms_target = self.target_net(next_states)
            next_q_atoms_target = next_q_atoms_target[range(self.batch_size), next_actions]

            Tz = rewards.unsqueeze(1) + (self.gamma ** self.n_steps) * self.support.unsqueeze(0) * (1 - dones.unsqueeze(1))
            Tz = Tz.clamp(self.Vmin, self.Vmax)
            b = (Tz - self.Vmin) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros(self.batch_size, self.num_atoms).to(self.device)
            offset = (torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size)
                      .long()
                      .unsqueeze(1)
                      .expand(self.batch_size, self.num_atoms)
                      ).to(self.device)

            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (next_q_atoms_target * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (next_q_atoms_target * (b - l.float())).view(-1))

        q_atoms = self.policy_net(states)
        q_atoms = q_atoms[range(self.batch_size), actions.squeeze(1)]

        log_q = torch.log(q_atoms + 1e-5)
        loss = -(m * log_q).sum(1)

        prios = loss + 1e-5
        loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

        if frame_idx % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, num_frames):
        state = self.env.reset()
        episode_reward = 0
        got_flag = False  # Initialize got_flag for the episode
        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)

            self.store_transition(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # Check if the flag was obtained in this step
            if info.get('flag_get', False):
                got_flag = True

            self.learn(frame_idx)

            if done:
                self.episode_counter += 1  # Increment episode counter
                print(f"Episode: {self.episode_counter}, Reward: {episode_reward}, Got Flag: {got_flag}")

                # Write to CSV file
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode_reward, got_flag])

                episode_reward = 0
                got_flag = False  # Reset for the next episode
                state = self.env.reset()
                self.n_step_buffer = []

                # Save model every 100 episodes
                if self.episode_counter % 100 == 0:
                    self.save_model()
                    print(f"Model saved at episode {self.episode_counter}")

                # Save checkpoint every 500 episodes
                if self.episode_counter % 500 == 0:
                    self.save_checkpoint()
                    print(f"Checkpoint saved at episode {self.episode_counter}")

            self.policy_net.reset_noise()
            self.target_net.reset_noise()

            if frame_idx % 1000 == 0:
                print(f"Frame: {frame_idx}")

    def save_model(self):
        model_path = os.path.join(self.save_path, 'latest_model.pth')
        torch.save({
            'episode': self.episode_counter,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.save_path, f'checkpoint_episode_{self.episode_counter}.pth')
        torch.save({
            'episode': self.episode_counter,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load_model(self):
        model_path = os.path.join(self.save_path, 'latest_model.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_counter = checkpoint.get('episode', 0)
            print(f"Loaded model from {model_path}, starting from episode {self.episode_counter}")
        else:
            print("No saved model found, starting from scratch.")

# Main function to run the training
def main():
    # Use the environment creation code provided
    env = create_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = Agent(env, input_channels, num_actions, device)

    num_frames = 1000000
    agent.train(num_frames)

if __name__ == "__main__":
    main()
