import os

# Set environment variables for CUDA and library paths
os.environ["LD_LIBRARY_PATH"] = "/cluster/home/andreksv/PycharmProjects/IDATT2502-NES-SM-ML/venv/lib64:/cluster/home/andreksv/PycharmProjects/IDATT2502-NES-SM-ML/venv/lib/python3.9/site-packages/nvidia/cusparse/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])

import csv
import hashlib
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.environment.environment import create_env
from src.ddqn.dqn_model import DQN

MODEL_PATH = "model/ddqn_model_episode_latest.pth"
TRAINING_RESULTS_DIR = "training_results"

class DDQNAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            replay_buffer_size=100000,
            batch_size=64,
            gamma=0.99,
            lr=0.0001,
            hard_update=5000,
            epsilon_start=0.1,
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize main and target networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-3)

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Create directory for training results
        os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.hard_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, env, num_episodes, save_interval=300):
        score_log = []
        start_episode = 1

        for episode in range(start_episode, num_episodes + 1):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
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
                print(f"Epsilon: {self.epsilon}")

            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}")
            score_log.append((episode, episode_reward))

            if episode % 10 == 0:
                model_filename = f"model/ddqn_model_episode_latest.pth"
                self.save_model(model_filename)

            # Save the model and replay buffer periodically
            if episode % save_interval == 0:
                model_filename = f"model/ddqn_model_episode_{episode}.pth"
                self.save_model(model_filename)
                self.save_scores(score_log, episode)
                score_log = []

    def save_scores(self, scores, episode):
        filename = os.path.join(TRAINING_RESULTS_DIR, f"training_results_episode_{episode}.csv")
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])
            writer.writerows(scores)
        print(f"Scores saved to {filename}")

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")

    def populate_replay_buffer(self, env, initial_size=10000):
        """Populate the replay buffer with random transitions"""
        state = env.reset()
        for _ in range(initial_size):
            action = random.randint(0, self.action_dim - 1)  # Random action for exploration
            next_state, reward, done, _ = env.step(action)
            self.store_transition(state, action, reward, next_state, done)
            state = next_state

            if done:
                state = env.reset()

        print(f"Replay buffer populated with {initial_size} transitions.")

def main():
    # Create the environment
    env = create_env()

    # Check CUDA and cuDNN availability
    print("CUDA available:", torch.cuda.is_available())
    print("cuDNN available:", torch.backends.cudnn.is_available())

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
    agent.populate_replay_buffer(env, initial_size=50000)

    # Train the agent
    num_episodes = 50000
    agent.train(env, num_episodes)

    # Save the trained model
    agent.save_model(MODEL_PATH)


if __name__ == "__main__":
    main()
