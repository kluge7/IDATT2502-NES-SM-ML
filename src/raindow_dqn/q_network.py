# q_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from noisy_linear import NoisyLinear


class QNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int,
        action_size: int,
        num_atoms: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        noisy: bool = True,
        dueling: bool = True,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.action_size = action_size
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, num_atoms)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc_input_dim = 64 * 7 * 7

        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 512)
            if noisy
            else nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, self.num_atoms)
            if noisy
            else nn.Linear(512, self.num_atoms),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 512)
            if noisy
            else nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, action_size * self.num_atoms)
            if noisy
            else nn.Linear(512, action_size * self.num_atoms),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)

        # Value and Advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.action_size, self.num_atoms)

        # Combine streams
        q_atoms = value + advantage - advantage.mean(1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)

        return q_dist

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
