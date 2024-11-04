import csv
import hashlib
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.environment.environment import create_env
from src.ddqn.dqn_model import DQN

MODEL_PATH = "model/ddqn_model.pth"
TRAINING_RESULTS_PATH = "training_results/training_results.csv"
REPLAY_BUFFER_PATH = "model/replay_buffer.npz"


class DDQNAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            replay_buffer_size=50000,
            batch_size=32,
            gamma=0.99,
            lr=0.0005,
            hard_update=2000,
            epsilon_start=0.01,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            update_counter=0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.hard_update = hard_update
        self.update_counter = update_counter

        # Set device
        self.device = torch.device("cpu")

        # Initialize main and target networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        if next(self.policy_net.parameters()).is_cuda:
            print("Using CUDA for model training.")
        else:
            print("Not using CUDA; training on CPU.")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-3)

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.load_replay_buffer()

    def select_action(self, state):
        if random.random() < self.epsilon:
            weights = [0.05, 0.2, 0.25, 0.2, 0.2, 0.1, 0.05]
            action = random.choices(range(self.action_dim), weights=weights)[0]
            return action
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(
                    self.device)  # Shape: (1, channels, height, width)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(
            self.device)  # Shape: (batch_size, channels, height, width)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute Q(s, a) with policy network
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using DDQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss and optimize the policy network
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Perform hard update on target network
        self.update_counter += 1
        if self.update_counter % self.hard_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, env, num_episodes):
        os.makedirs("model", exist_ok=True)
        os.makedirs("training_results", exist_ok=True)

        # Check if the CSV already exists and load the last episode number if it does
        start_episode = 1
        if os.path.isfile(TRAINING_RESULTS_PATH):
            with open(TRAINING_RESULTS_PATH, mode="r") as file:
                last_line = None
                for last_line in file:
                    pass  # Iterate to the end of the file
                if last_line is not None and last_line.strip():
                    start_episode = int(last_line.split(",")[0]) + 1  # Start from the next episode

        with open(TRAINING_RESULTS_PATH, mode="a", newline="") as file:  # Open in append mode
            writer = csv.writer(file)
            # Write header only if file is new
            if start_episode == 1:
                writer.writerow(["Episode", "Reward"])

            for episode in range(num_episodes):
                state = env.reset()  # State shape: (channels, height, width)
                done = False
                episode_reward = 0

                while not done:
                    env.render()
                    action = self.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward

                    self.store_transition(state, action, reward, next_state, done)
                    self.update()

                    state = next_state

                # Decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                    self.epsilon = max(self.epsilon, self.epsilon_min)
                    print(f"Epsilon after decay: {self.epsilon}")

                print(f"Episode {start_episode + episode}/{num_episodes}, Reward: {episode_reward}")

                # Append the episode reward to the CSV file with the correct episode number
                writer.writerow([start_episode + episode, episode_reward])

                # Save the model every 10 episodes
                if (start_episode + episode) % 10 == 0:
                    self.save_model(MODEL_PATH)

                # Save the replay buffer every 50 episodes
                if (start_episode + episode) % 50 == 0:
                    self.save_replay_buffer()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")
        checksum = get_model_checksum(self.policy_net)
        print(f"Checksum of saved model: {checksum}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")
        checksum = get_model_checksum(self.policy_net)
        print(f"Checksum of saved model: {checksum}")

    def save_replay_buffer(self):
        """Saves the replay buffer to a file."""
        buffer_data = {
            "state": [t[0] for t in self.replay_buffer],
            "action": [t[1] for t in self.replay_buffer],
            "reward": [t[2] for t in self.replay_buffer],
            "next_state": [t[3] for t in self.replay_buffer],
            "done": [t[4] for t in self.replay_buffer]
        }
        np.savez_compressed(REPLAY_BUFFER_PATH, **buffer_data)
        print(f"Replay buffer saved to {REPLAY_BUFFER_PATH}")

    def load_replay_buffer(self):
        """Loads the replay buffer from a file if available."""
        if os.path.exists(REPLAY_BUFFER_PATH):
            data = np.load(REPLAY_BUFFER_PATH, allow_pickle=True)
            self.replay_buffer = deque(
                zip(data["state"], data["action"], data["reward"], data["next_state"], data["done"]),
                maxlen=self.replay_buffer.maxlen
            )
            print(f"Replay buffer loaded from {REPLAY_BUFFER_PATH}")


    def populate_replay_buffer(self, env, initial_size):
        state = env.reset()
        for _ in range(initial_size):
            env.render()
            action = random.randint(0, self.action_dim - 1)
            next_state, reward, done, _ = env.step(action)
            self.store_transition(state, action, reward, next_state, done)
            if done:
                state = env.reset()
            else:
                state = next_state

    


def get_model_checksum(model):
    """Returns a checksum of the model's parameters."""
    md5 = hashlib.md5()
    for param in model.parameters():
        md5.update(param.detach().cpu().numpy().tobytes())
    return md5.hexdigest()


def main():
    # Create the environment
    env = create_env()

    # Get input dimensions and number of actions
    in_dim = env.observation_space.shape  # Expected shape: (channels, height, width)
    num_actions = env.action_space.n

    # Initialize the DDQN agent
    agent = DDQNAgent(state_dim=in_dim, action_dim=num_actions)

    if os.path.exists(MODEL_PATH):
        agent.load_model(MODEL_PATH)
    else:
        print("No pre-trained model found")

    # Populate the replay buffer with random transitions
    agent.populate_replay_buffer(env, initial_size=10000)

    # Train the agent
    num_episodes = 50000  # Adjust as needed
    agent.train(env, num_episodes)

    # Save the trained model
    agent.save_model(MODEL_PATH)


if __name__ == "__main__":
    main()
