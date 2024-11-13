import torch
from torch import nn
from torch.optim import Adam

from src.ppo.cnn_network import CNNNetwork
from src.supervised.utils import dataset_utils


class CriticCNNNetwork(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        # Output a single value for binary classification (0 or 1)
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),  # Sigmoid for binary classification output
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def generate_random_data(batch_size, in_dim):
    random_data = torch.rand(batch_size, *in_dim)
    return random_data


def train_for_ppo_critic():
    # load inn model
    # load inn all treningsdata

    # load inn dårlig data??????
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervised_data, labels = dataset_utils.load_dataset()
    random_data = generate_random_data(
        batch_size=supervised_data.size(0), in_dim=(4, 84, 84)
    )

    supervised_model = CNNNetwork(
        in_dim=(4, 84, 84), out_dim=12
    )  # load inn fra allerede eksiterende. ikke gjør slik det er nå
    critic = CriticCNNNetwork(in_dim=(4, 84, 84))

    supervised_model.load_state_dict(
        torch.load(
            "src/supervised/model/action-prediction/1-1/ActionPredictionModel.pth",
            map_location=device,
        )
    )
    print(f"model features: {supervised_model.features}")
    # self.critic.load_state_dict(torch.load(critic_path))

    # Load pretrained weights from supervised model to critic and freeze CNN layers
    critic.features.load_state_dict(supervised_model.features.state_dict())
    for param in critic.features.parameters():
        param.requires_grad = False

    print(f"critic features: {critic.features}")

    # Set up criterion and optimizer for binary classification
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for 0 and 1 outputs
    optimizer = Adam(critic.fc.parameters(), lr=1e-3)  # Train only the fc layer

    supervised_labels = torch.ones(
        (supervised_data.size(0), 1)
    )  # Labels 1 for supervised data

    # Concatenate data and labels
    random_data = generate_random_data(
        batch_size=supervised_data.size(0), in_dim=(4, 84, 84)
    )
    random_labels = torch.zeros((random_data.size(0), 1))  # Labels 0 for random data

    # Concatenate supervised and random data
    data = torch.cat((supervised_data, random_data), dim=0)
    labels = torch.cat((supervised_labels, random_labels), dim=0)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        critic.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = critic(data)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


train_for_ppo_critic()
