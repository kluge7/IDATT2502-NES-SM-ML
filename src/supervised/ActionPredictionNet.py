import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from torch.utils.data import DataLoader, Dataset

from src.supervised.utils import dataset_utils
from src.utils import get_unique_filename

action_to_index = {
    tuple(sorted(action)): i for i, action in enumerate(COMPLEX_MOVEMENT)
}
index_to_action = {idx: action for idx, action in enumerate(COMPLEX_MOVEMENT)}


class FrameSequenceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


class ActionPredictionNet(nn.Module):
    def __init__(self, input_channels: int, action_size: int):
        super().__init__()
        print(input)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, action_size)
        )

    def forward(self, x):
        """Defines the forward pass of the Q-network.

        Args:
            x (torch.Tensor): Input tensor representing the state, with dimensions
                (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor representing action-value estimates for each
            possible action, with dimensions (batch_size, action_size).

        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ActionPredictionModel:
    def __init__(
        self, input_channels: int = 3, action_size: int = len(action_to_index)
    ):
        self.model = ActionPredictionNet(input_channels, action_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)

    def save_model(self, save_path):
        """Save the model state to the given path."""
        torch.save(self.model.state_dict(), save_path)

    def train(self, train_loader, epochs=10):
        training_result_path = "src/supervised/training_results/action-prediction"
        os.makedirs("src/supervised/model/action-prediction/", exist_ok=True)
        os.makedirs(
            "src/supervised/model/action-prediction/checkpoints/", exist_ok=True
        )
        os.makedirs(training_result_path, exist_ok=True)

        # CSV file to log the training results
        output_csv = "training_results.csv"
        training_output_file = get_unique_filename(training_result_path, output_csv)

        with open(training_output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss"])

            # Set the model to training mode
            self.model.train()

            for epoch in range(epochs):
                epoch_loss = 0
                for inputs, labels in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()

                average_epoch_loss = epoch_loss / len(train_loader)

                self.save_model(
                    "src/supervised/model/action-prediction/ActionPredictionModel.pth"
                )

                if epoch % 100 == 0:
                    save_path = f"src/supervised/model/action-prediction/checkpoints/ActionPredictionModel-epoch{epoch}.pth"
                    self.save_model(save_path)

                writer.writerow([epoch + 1, average_epoch_loss])
                file.flush()
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_epoch_loss:.4f}")


def main():
    images, labels = dataset_utils.load_dataset()

    labels = [action_to_index[tuple(sorted(label))] for label in labels]
    y_train_tensor = torch.tensor(labels, dtype=torch.long)
    images = images.squeeze(2)
    train_dataset = FrameSequenceDataset(images, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    input_channels = 4
    action_size = len(action_to_index)

    model = ActionPredictionModel(input_channels, action_size)
    model.train(train_loader, epochs=1)


if __name__ == "__main__":
    main()
