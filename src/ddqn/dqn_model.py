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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute the output size after conv layers
        conv_out_size = self._get_conv_out(in_dim)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),  # Use dynamically calculated conv_out_size
            nn.ReLU(),
            nn.Linear(512, num_actions),  # Output layer with 'num_actions' outputs
        )

    def _get_conv_out(self, shape):
        """Passes a dummy input through conv layers to get the output size."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)  # Create a dummy input with the specified shape
            output = self.conv(dummy_input)  # Pass through conv layers
            return int(output.view(1, -1).size(1))  # Flatten and get the total size

    def forward(self, state):
        """Forward pass through the DQN.

        Args:
            state (torch.Tensor): The input state (batch of images).

        Returns:
            torch.Tensor: The output of the network.
        """
        x = state.clone().detach().to(dtype=torch.float, device=self.device)
        x = self.conv(x)  # Pass through conv layers
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)  # Pass through fully connected layers
