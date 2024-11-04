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
TRAINING_RESULTS_DIR = "training_results"
REPLAY_BUFFER_PATH = "model/replay_buffer.npz"


class DDQNAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            replay_buffer_size=10000,
            batch_size=32,
            gamma=0.99,
            lr=0.0005,
            hard_update=5000,
            epsilon_start=1.0,
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
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-3)

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.load_replay_buffer()

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

    def train(self, env, num_episodes, save_interval=100):
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

            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}")
            score_log.append((episode, episode_reward))

            # Save the model and replay buffer periodically
            if episode % save_interval == 0:
                model_filename = f"model/ddqn_model_episode_{episode}.pth"
                self.save_model(model_filename)
                self.save_replay_buffer()
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

    def save_replay_buffer(self):
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
        if os.path.exists(REPLAY_BUFFER_PATH):
            data = np.load(REPLAY_BUFFER_PATH, allow_pickle=True)
            self.replay_buffer = deque(
                zip(data["state"], data["action"], data["reward"], data["next_state"], data["done"]),
                maxlen=self.replay_buffer.maxlen
            )
            print(f"Replay buffer loaded from {REPLAY_BUFFER_PATH}")

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
    if not os.path.exists(REPLAY_BUFFER_PATH):
        agent.populate_replay_buffer(env, initial_size=10000)

    # Train the agent
    num_episodes = 50000
    agent.train(env, num_episodes)

    # Save the trained model
    agent.save_model(MODEL_PATH)


if __name__ == "__main__":
    main()
