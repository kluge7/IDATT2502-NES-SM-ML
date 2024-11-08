import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_channels: int, action_size: int):
        super(QNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels, 32, kernel_size=8, stride=4
            ),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (64, 7, 7)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
