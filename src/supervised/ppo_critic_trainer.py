import csv
import os

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
    # Load in model, training data, and initialize necessary components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervised_data_wins, _ = dataset_utils.load_dataset()
    supervised_data_fails, _ = dataset_utils.load_dataset(is_fails=True)

    supervised_model = CNNNetwork(in_dim=(4, 84, 84), out_dim=12)
    critic = CriticCNNNetwork(in_dim=(4, 84, 84))

    # Load pretrained model
    supervised_model.load_state_dict(
        torch.load(
            "src/supervised/model/action-prediction/1-1/ActionPredictionModel.pth",
            map_location=device,
        )
    )

    # Load weights into critic model and freeze layers
    critic.features.load_state_dict(supervised_model.features.state_dict())
    for param in critic.features.parameters():
        param.requires_grad = False

    # Set up criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = Adam(critic.fc.parameters(), lr=1e-3)

    # Labels for supervised and random data
    supervised_labels_wins = torch.ones((supervised_data_wins.size(0), 1))
    supervised_labels_fails = torch.zeros((supervised_data_fails.size(0), 1))

    # Concatenate data and labels
    data = torch.cat((supervised_data_wins, supervised_data_fails), dim=0)
    labels = torch.cat((supervised_labels_wins, supervised_labels_fails), dim=0)

    # Prepare CSV file for logging losses
    training_result_path = "src/supervised/training_results/ppo_critic"
    os.makedirs(training_result_path, exist_ok=True)
    output_csv = "ppo_critic_training_results.csv"
    training_output_file = os.path.join(training_result_path, output_csv)

    with open(training_output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])

        # Training loop
        num_epochs = 1000
        for epoch in range(num_epochs):
            critic.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = critic(data)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Log the loss for this epoch
            writer.writerow([epoch + 1, loss.item()])
            file.flush()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

            # Save checkpoints
            if epoch % 100 == 0:
                checkpoint_path = f"src/supervised/model/action-prediction/ppo-critic/checkpoints/trained_critic-e{epoch}.pth"
                torch.save(critic.state_dict(), checkpoint_path)

    # Final model save
    torch.save(
        critic.state_dict(),
        "src/supervised/model/action-prediction/ppo-critic/trained_critic.pth",
    )


train_for_ppo_critic()
