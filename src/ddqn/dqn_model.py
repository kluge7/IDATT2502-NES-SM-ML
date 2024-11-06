import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_dim, num_actions):
        super(DQN, self).__init__()
        
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

        # Advantage and value layers
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            conv_out = self.conv(dummy_input)
            return int(conv_out.numel() / conv_out.size(0))

    def forward(self, x):
        x = x.float()  # Ensure input is float32
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten

        adv = self.fc_adv(conv_out)
        val = self.fc_val(conv_out)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q
