import csv
import hashlib
import os
import random
from collections import deque

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
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.95,
        lr=0.00025,
        tau=0.005,
        epsilon_start=0.05,
        epsilon_min=0.05,
        epsilon_decay=0.99
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay


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
        self.update_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state = state.to(self.device).unsqueeze(0)  # Add batch dimension
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
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute Q(s, a) with policy network
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using DDQN
        with torch.no_grad():
            next_action_indices = self.policy_net(next_states).argmax(dim=1)
            next_q_values = (
                self.target_net(next_states)
                .gather(1, next_action_indices.unsqueeze(1))
                .squeeze(1)
            )
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and optimize the policy network
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Perform soft update on target network
        self.soft_update_target_network()

    def soft_update_target_network(self):
        """Softly updates the target network by blending it with the policy network."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def train(
        self, env, num_episodes
    ):
        os.makedirs("model", exist_ok=True)
        os.makedirs("training_results", exist_ok=True)

        with open(TRAINING_RESULTS_PATH, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])

        for episode in range(num_episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            episode_reward = 0

            while not done:
                env.render()
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                episode_reward += reward

                self.store_transition(state, action, reward, next_state, done)
                self.update()

                state = next_state

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                print(f"Epsilon after decay: {self.epsilon}")

            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

            # Save the episode reward to the CSV file
            with open(TRAINING_RESULTS_PATH, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([episode + 1, episode_reward])

            # Save the model every 2 episodes
            if (episode + 1) % 2 == 0:
                self.save_model(MODEL_PATH)

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
    in_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # Initialize the DDQN agent
    agent = DDQNAgent(state_dim=in_dim, action_dim=num_actions)

    if os.path.exists(MODEL_PATH):
        agent.load_model(MODEL_PATH)
    else:
        print("No pre-trained model found")

    # Train the agent
    num_episodes = 10000  # You can change this based on how long you want to train
    agent.train(env, num_episodes)

    # Save the trained model
    agent.save_model(MODEL_PATH)


if __name__ == "__main__":
    main()
