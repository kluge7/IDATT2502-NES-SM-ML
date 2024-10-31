import gym_super_mario_bros
import torch
import torch.nn.functional as functional
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch import nn, optim

train_loader = []


class CNNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 8)  # Adjust according to the number of actions

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = self.pool(functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Initialize the model, loss function, and optimizer


model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Training loop
num_epochs = 10

for _epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        # Saving the model if used in different environment
        torch.save(model.state_dict(), "super_mario.pth")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Super Mario Miljøet


env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for _step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
