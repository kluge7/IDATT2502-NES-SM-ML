from torch import nn
import torch


class CNNNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = nn.Sequential(
            nn.Conv2d(in_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(), nn.Linear(512, out_dim)
        )
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
