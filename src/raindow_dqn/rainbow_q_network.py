# q_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from noisy_linear import NoisyLinear


class RainbowQNetwork(nn.Module):
    """Rainbow DQN with dueling architecture, noisy layers, and distributional outputs."""

    def __init__(self, input_channels, action_size, atom_size, support):
        super(RainbowQNetwork, self).__init__()
        self.support = support
        self.action_size = action_size
        self.atom_size = atom_size

        self.feature_layer = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (64, 7, 7)
            nn.ReLU(),
            nn.Flatten()
        )
        self.noisy_value1 = NoisyLinear(64 * 7 * 7, 512)
        self.noisy_value2 = NoisyLinear(512, self.atom_size)

        self.noisy_advantage1 = NoisyLinear(64 * 7 * 7, 512)
        self.noisy_advantage2 = NoisyLinear(512, action_size * self.atom_size)

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x):
        x = self.feature_layer(x)
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        value = value.view(-1, 1, self.atom_size)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        advantage = advantage.view(-1, self.action_size, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=2)
        dist = dist.clamp(min=1e-3)  # To avoid log(0)
        return dist

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()