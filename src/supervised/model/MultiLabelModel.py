import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset

from src.supervised.utils import dataset_utils


class MultiLabelNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Define a simple feed-forward network
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),  # Hidden layer
            nn.ReLU(),
            nn.Linear(128, output_size),  # Output layer
        )

    def forward(self, x):
        # Apply sigmoid activation for multi-label binary outputs
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))


class MultiLabelModel:
    def __init__(self, input_size, output_size):
        self.model = MultiLabelNet(input_size, output_size)
        self.criterion = nn.BCEWithLogitsLoss()  # Combines Sigmoid and BCELoss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader, epochs=10):
        self.model.train()
        for __epoch in range(epochs):
            epoch_loss = 0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

    """
    def evaluate(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            all_outputs = []
            all_labels = []
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                all_outputs.append(outputs)
                all_labels.append(labels)

        # Convert outputs to binary predictions and calculate accuracy
        outputs = torch.cat(all_outputs)
        labels = torch.cat(all_labels)
        binary_preds = (outputs > 0.5).float()

        accuracy = (binary_preds == labels).all(dim=1).float().mean().item()
    """


images, labels = dataset_utils.load_dataset()

possible_actions = ["A", "up", "left", "B", "start", "right", "down", "NOOP"]

images_train, images_test, labels_train, labels_test = dataset_utils.train_test_spit(
    images, labels, 0.7
)
input_size = images_train[0].numel()
output_size = 8

mlb = MultiLabelBinarizer(classes=possible_actions)
encoded_labels_train = mlb.fit_transform(labels_train)
encoded_labels_test = mlb.transform(labels_test)

y_train_tensor = torch.tensor(encoded_labels_train, dtype=torch.float32)
y_test_tensor = torch.tensor(encoded_labels_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(images_train, y_train_tensor), batch_size=32, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(images_test, y_test_tensor), batch_size=32, shuffle=False
)

model = MultiLabelModel(input_size, output_size)
model.train(train_loader, epochs=50)

model.evaluate(test_loader)
