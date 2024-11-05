import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset

from src.supervised.utils import dataset_utils


class MultiLabelNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 60 * 64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class MultiLabelModel:
    def __init__(self, num_classes):
        self.model = MultiLabelNet(num_classes)
        self.criterion = nn.BCEWithLogitsLoss()  # Combines Sigmoid and BCELoss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader, epochs=10):
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
            print(f"Epoch: {epoch}, Loss: {epoch_loss}")

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

                # Accumulate predictions and true labels for metric calculation
                all_outputs.append(outputs)
                all_labels.append(labels)

        # Convert outputs and labels to tensors
        outputs = torch.cat(all_outputs)
        labels = torch.cat(all_labels)

        # Apply sigmoid to outputs for probability conversion
        outputs = torch.sigmoid(outputs)
        binary_preds = (outputs > 0.5).float()

        # Calculate overall metrics
        precision = precision_score(labels.cpu(), binary_preds.cpu(), average="samples")
        recall = recall_score(labels.cpu(), binary_preds.cpu(), average="samples")
        f1 = f1_score(labels.cpu(), binary_preds.cpu(), average="samples")

        # Print evaluation results
        print(f"Test Loss: {total_loss / len(test_loader):.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        return precision, recall, f1


images, labels = dataset_utils.load_dataset()

possible_actions = ["A", "up", "left", "B", "start", "right", "down", "NOOP"]

images_train, images_test, labels_train, labels_test = dataset_utils.train_test_spit(
    images, labels, 0.7
)
input_size = images_train[0].numel()
num_classes = 8

mlb = MultiLabelBinarizer(classes=possible_actions)
encoded_labels_train = mlb.fit_transform(labels_train)
encoded_labels_test = mlb.transform(labels_test)

y_train_tensor = torch.tensor(encoded_labels_train, dtype=torch.float32)
y_test_tensor = torch.tensor(encoded_labels_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(images_train, y_train_tensor), batch_size=16, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(images_test, y_test_tensor), batch_size=16, shuffle=False
)

model = MultiLabelModel(num_classes)

model.train(train_loader, epochs=50)

model.evaluate(test_loader)
