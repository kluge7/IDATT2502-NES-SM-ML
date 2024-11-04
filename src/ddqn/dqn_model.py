import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_dim, num_actions):
        """Initializes the DQN with specified input dimensions and number of actions.

        Args:
            in_dim (tuple): Input dimensions (channels, height, width).
            num_actions (int): Number of actions for the output layer.
        """
        super(DQN, self).__init__()

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim[0], 32, kernel_size=8, stride=4),  # Input channels from in_dim
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Compute the output size after conv layers
        conv_out_size = self._get_conv_out(in_dim)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_actions),
        )

    def _get_conv_out(self, shape):
        """Passes a dummy input through conv layers to get the output size."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            conv_out = self.conv(dummy_input)
            return int(conv_out.numel() / conv_out.size(0))

    def forward(self, x):
        """Forward pass through the DQN.

        Args:
            x (torch.Tensor): The input state (batch of images).

        Returns:
            torch.Tensor: The output of the network.
        """
        x = x.float()  # Ensure the input is float32
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten
        return self.fc(conv_out)
