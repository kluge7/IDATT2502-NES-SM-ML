import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset

from src.supervised.utils import dataset_utils
from src.utils import get_unique_filename


class FrameSequenceDataset(Dataset):
    def __init__(self, frames, labels, seq_length):
        self.frames = frames
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        # Number of sequences we can create
        return len(self.frames) - self.seq_length + 1

    def __getitem__(self, idx):
        # Extract a sequence of frames and the corresponding label
        frame_seq = self.frames[idx : idx + self.seq_length]
        label = self.labels[
            idx + self.seq_length - 1
        ]  # Label corresponds to the last frame in the sequence
        return frame_seq, label


class MultiLabelNet(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=64, num_lstm_layers=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Adaptive pooling to control the output size for LSTM input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((15, 16))  # Adjust size as needed
        self.flatten_size = 64 * 15 * 16  # Updated based on adaptive pooling output

        self.lstm = nn.LSTM(
            input_size=self.flatten_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()

        # Combine batch and sequence dimensions for CNN processing
        x = x.view(batch_size * seq_length, channels, height, width)

        # Convolutional layers with pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Adaptive pooling to standardize output size
        x = self.adaptive_pool(x)

        # Flatten to prepare for LSTM input
        x = x.view(
            batch_size, seq_length, -1
        )  # Reshape to (batch_size, seq_length, flatten_size)

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Use the output from the last LSTM cell in the sequence
        x = lstm_out[:, -1, :]  # Last LSTM output in the sequence for each batch

        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class MultiLabelModel:
    def __init__(self, num_classes, lstm_hidden_size=64, num_lstm_layers=1):
        self.model = MultiLabelNet(num_classes, lstm_hidden_size, num_lstm_layers)
        self.criterion = nn.BCEWithLogitsLoss()  # Combines Sigmoid and BCELoss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)

    def train(self, train_loader, epochs=10):
        training_result_path = "src/supervised/training_results/"
        os.makedirs("src/supervised/model/", exist_ok=True)
        os.makedirs("src/supervised/model/checkpoints/", exist_ok=True)
        os.makedirs(training_result_path, exist_ok=True)

        output_csv = "training_results.csv"
        training_output_file = get_unique_filename(training_result_path, output_csv)

        with open(training_output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss"])

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

                # Calculate average loss for the epoch
                average_epoch_loss = epoch_loss / len(train_loader)

                # Save model after each epoch and every 100 epochs in a checkpoint
                self.save_model("src/supervised/model/MultiLabelLSTM.pth")
                if epoch % 100 == 0:
                    save_path = f"src/supervised/model/checkpoints/MultiLabelLSTM-epoch{epoch}.pth"
                    self.save_model(save_path)

                # Write the epoch and loss to the CSV file
                writer.writerow([epoch + 1, average_epoch_loss])
                file.flush()

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                all_outputs.append(outputs)
                all_labels.append(labels)

        outputs = torch.cat(all_outputs)
        labels = torch.cat(all_labels)

        outputs = torch.sigmoid(outputs)
        binary_preds = (outputs > 0.5).float()

        precision = precision_score(labels.cpu(), binary_preds.cpu(), average="samples")
        recall = recall_score(labels.cpu(), binary_preds.cpu(), average="samples")
        f1 = f1_score(labels.cpu(), binary_preds.cpu(), average="samples")

        print(f"Test Loss: {total_loss / len(test_loader):.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        return precision, recall, f1

    def save_model(self, file_path):
        """Save the model parameters to a file."""
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load the model parameters from a file."""
        self.model.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")


def main():
    images, labels = dataset_utils.load_dataset()
    possible_actions = ["A", "up", "left", "B", "start", "right", "down", "NOOP"]

    mlb = MultiLabelBinarizer(classes=possible_actions)
    encoded_labels_train = mlb.fit_transform(labels)
    y_train_tensor = torch.tensor(encoded_labels_train, dtype=torch.float32)

    # Generate datasets with sequences
    seq_length = 4
    train_dataset = FrameSequenceDataset(images, y_train_tensor, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = MultiLabelModel(num_classes=8)
    model.train(train_loader, epochs=50)


if __name__ == "__main__":
    main()
