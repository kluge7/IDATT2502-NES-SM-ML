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

class DDQNAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            replay_buffer_size=50000,
            batch_size=256,
            gamma=0.99,
            lr=1E-4,
            hard_update=50,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            update_counter=0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.hard_update = hard_update
        self.update_counter = update_counter

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize main and target networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

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
        if len(self.replay_buffer) < max(2000, self.batch_size):
            return 0

        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute Q(s, a) with policy network
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using Double Q-learning
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.hard_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def train(self, env, num_episodes):
        os.makedirs("model", exist_ok=True)
        os.makedirs("training_results", exist_ok=True)

        start_episode = 1
        if os.path.isfile(TRAINING_RESULTS_PATH):
            with open(TRAINING_RESULTS_PATH, mode="r") as file:
                last_line = None
                for last_line in file:
                    pass
                if last_line and not last_line.startswith("Episode"):
                    start_episode = int(last_line.split(",")[0]) + 1

        with open(TRAINING_RESULTS_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)
            if start_episode == 1:
                writer.writerow(["Episode", "Reward", "Average Loss", "Completed"])

            for episode in range(num_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                losses = []
                completed = False

                while not done:
                    action = self.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    self.store_transition(state, action, reward, next_state, done)

                    # Update and collect loss for this step
                    loss = self.update()
                    losses.append(loss)

                    # Check if the environment was completed
                    if done and "flag_get" in info:
                        completed = bool(info["flag_get"])

                    state = next_state

                avg_loss = np.mean(losses) if losses else 0
                print(f"Episode {start_episode + episode}/{num_episodes}, Reward: {episode_reward}, Avg Loss: {avg_loss:.4f}, Completed: {completed}")
                writer.writerow([start_episode + episode, episode_reward, avg_loss, completed])

                # Decay epsilon
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                print(f"Epsilon after decay: {self.epsilon}")

                if (start_episode + episode) % 100 == 0:
                    self.save_model(MODEL_PATH)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")

    def populate_replay_buffer(self, env, initial_size):
        state = env.reset()
        for _ in range(initial_size):
            action = random.randint(0, self.action_dim - 1)
            next_state, reward, done, _ = env.step(action)
            self.store_transition(state, action, reward, next_state, done)
            state = env.reset() if done else next_state

def main():
    env = create_env()
    in_dim = env.observation_space.shape
    num_actions = env.action_space.n

    agent = DDQNAgent(state_dim=in_dim, action_dim=num_actions)

    if os.path.exists(MODEL_PATH):
        agent.load_model(MODEL_PATH)
    else:
        print("No pre-trained model found")

    agent.populate_replay_buffer(env, initial_size=10000)
    num_episodes = 50000
    agent.train(env, num_episodes)
    agent.save_model(MODEL_PATH)

if __name__ == "__main__":
    main()
