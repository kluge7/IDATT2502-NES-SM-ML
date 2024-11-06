import torch
import torch.nn as nn
import torch.nn.functional as F

from src.supervised.utils.dataset_utils import load_dataset, preprocess_frame

action_map = {
    0: 0,  # NOOP
    4: 1,  # right
    132: 2,  # right + A
    20: 3,  # right + B
    148: 4,  # right + A + B
    128: 5,  # A
    8: 6,  # left
    136: 7,  # left + A
    24: 8,  # left + B
    152: 9,  # left + A + B
    2: 10,  # down
}


def decode_action(action_int):
    actions = {
        128: "A",
        64: "up",
        32: "left",
        16: "B",
        8: "start",
        4: "right",
        2: "down",
        1: "select",
    }
    active_actions = []
    for bit_value, action_name in actions.items():
        if action_int & bit_value:
            active_actions.append(action_name)
    return active_actions


class MarioCNN(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 60 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 60 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


num_classes = 256  # Total number of possible actions (0-255)
model = MarioCNN(num_classes)


import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class MarioDataset(Dataset):
    def __init__(self, images, actions):
        self.images = images
        self.actions = actions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        action = self.actions[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(
            action, dtype=torch.long
        )


images, labels = load_dataset()


labels = torch.tensor(labels)


dataset = MarioDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(3):
    running_loss = 0.0
    for images, actions in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")


torch.save(model.state_dict(), "mario_cnn_model_final.pth")

import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Initialize the Super Mario environment
env = gym.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = MarioCNN(num_classes)
# model.load_state_dict(torch.load("mario_cnn_model_final.pth", weights_only=True))
model.eval()


state = env.reset()
done = False

while not done:
    env.render()

    state_preprocessed = preprocess_frame(state)

    with torch.no_grad():
        action_prob = model(state_preprocessed)
        predicted_action = torch.argmax(action_prob).item()

    decoded_actions = decode_action(predicted_action)
    print(f"Predicted action {predicted_action} corresponds to: {decoded_actions}")

    action = action_map.get(predicted_action)

    state, reward, done, info = env.step(action)

env.close()
