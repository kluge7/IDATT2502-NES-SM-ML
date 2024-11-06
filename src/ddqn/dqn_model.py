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

        # Compute the output size after convolutional layers
        conv_out_size = self._get_conv_out(in_dim)

        # Separate streams for advantage and value
        self.fc = nn.Linear(conv_out_size, 512)
        self.q = nn.Linear(512, num_actions)
        self.v = nn.Linear(512, 1)

        self.apply(self._init_weights)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            conv_out = self.conv(dummy_input)
            return int(conv_out.numel() / conv_out.size(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.float()  # Ensure input is float32
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten
        x = torch.relu(self.fc(conv_out))
        adv = self.q(x)
        val = self.v(x)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q
