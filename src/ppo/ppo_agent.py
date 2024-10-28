import csv
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from src.environment.environment import create_env
from src.ppo.cnn_network import CNNNetwork

ACTOR_PATH = "networks/actor.pth"
CRITIC_PATH = "networks/critic.pth"


class PPOAgent:
    def __init__(
        self,
        in_dim,
        num_actions,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        lam=0.95,
        entropy_coef=0.01,
    ):
        """Initialize the PPO Agent with separate actor and critic networks.

        Args:
            in_dim: Tuple, input dimensions for the CNN.
            num_actions: Integer, number of possible actions.
            lr: Float, learning rate for the optimizer.
            gamma: Float, discount factor for future rewards.
            epsilon: Float, PPO clipping parameter.
            lam: Float, GAE lambda parameter.
            entropy_coef: Float, coefficient for entropy regularization.

        """
        self.gamma = gamma  # Discount factor for rewards
        self.epsilon = epsilon  # Clipping parameter for PPO
        self.lam = lam  # Lambda for GAE advantage calculation
        self.entropy_coef = entropy_coef  # Entropy coefficient for exploration

        # Detect if GPU is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")

        # Initialize the actor and critic networks
        self.actor = CNNNetwork(in_dim, out_dim=num_actions).to(
            self.device
        )  # Outputs action logits
        self.critic = CNNNetwork(in_dim, out_dim=1).to(
            self.device
        )  # Outputs value estimate

        # Separate optimizers for actor and critic networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
        """Select an action based on the current policy and return log_prob and value.

        Args:
            state: Torch tensor, the current state observation.

        Returns:
            action: The selected action (integer).
            log_prob: Log probability of the selected action.
            value: Value estimate of the current state.

        """
        state = state.to(
            self.device
        )  # Ensure the state tensor is on the correct device
        with torch.no_grad():
            logits = self.actor(state)
            value = self.critic(state).squeeze(-1)  # V(s)
            probs = F.softmax(logits, dim=-1)  # \(\pi(a|s)\)
            dist = Categorical(probs)
            action = dist.sample()  # Sample action a ~ \(\pi(a|s)\)
            log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def compute_gae(self, rewards, values, next_value, dones):
        """Calculate advantages and returns using Generalized Advantage Estimation (GAE).

        Formula:
            delta_t = reward + gamma * V(s') - V(s)
            advantage_t = delta_t + gamma * lambda * advantage_t+1

        Args:
            rewards: List of rewards collected in an episode.
            values: List of state values collected in an episode.
            next_value: Float, value of the next state.
            dones: List of booleans indicating episode termination.

        Returns:
            advantages: Torch tensor, advantage estimates for each timestep.
            returns: Torch tensor, target returns for each timestep.

        """
        values = values + [next_value]
        gae = 0
        advantages = []
        returns = []

        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            # GAE formula for calculating advantage estimate
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return torch.tensor(advantages, device=self.device), torch.tensor(
            returns, device=self.device
        )

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update the actor and critic networks using PPO loss.

        PPO Loss Formula:
            L_clip = E[min(r_t(theta) * A, clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A)]
            Critic loss = (returns - V(s))^2
            Total loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

        Args:
            states: Torch tensor, batch of states.
            actions: Torch tensor, batch of actions taken.
            old_log_probs: Torch tensor, log probabilities of actions taken.
            returns: Torch tensor, target returns.
            advantages: Torch tensor, advantage estimates.

        """
        for _ in range(4):  # Multiple epochs for each batch
            # Actor update
            logits = self.actor(states)
            dist = Categorical(F.softmax(logits, dim=-1))
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Calculate PPO clipped objective
            ratios = torch.exp(
                log_probs - old_log_probs
            )  # r_t(theta) = exp(log_pi - log_pi_old)
            surr1 = ratios * advantages  # r_t * A
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()  # PPO clipped objective

            # Add entropy to encourage exploration
            actor_loss -= self.entropy_coef * entropy

            # Update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic update
            values = self.critic(states).squeeze(-1)
            critic_loss = F.mse_loss(
                values, returns
            )  # Mean Squared Error for value function

            # Update critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def save(self, path="ppo_agent"):
        """Saves the actor and critic networks to .pth files.

        Args:
            path: Base path for saving the model weights.

        """
        os.makedirs(path, exist_ok=True)  # Create directory if it doesn't exist
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        print(f"Model saved to {path}")

    def train(self, env, num_episodes, save_path="training_results.csv"):
        """Train the agent using PPO in the specified environment.

        Args:
            env: The wrapped environment to train the agent in.
            num_episodes: Integer, number of episodes to train the agent.
            save_path: Path to save the training results.

        """
        # Initialize or reset the save file for results
        with open(save_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])  # Write headers

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
            episode_reward = 0

            while not done:
                # Ensure the state tensor is on the correct device
                state_tensor = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                action, log_prob, value = self.select_action(state_tensor)

                # Interact with environment
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                # Store experience for this episode
                states.append(state_tensor)
                actions.append(torch.tensor(action, device=self.device))
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)

                state = next_state

            # Compute advantages and returns
            next_state_tensor = (
                torch.tensor(next_state, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                next_value = self.critic(next_state_tensor).squeeze(-1)

            advantages, returns = self.compute_gae(rewards, values, next_value, dones)

            # Convert episode data to torch tensors for batch processing
            states = torch.cat(states)
            actions = torch.stack(actions)
            old_log_probs = torch.stack(log_probs)
            returns = returns.detach()
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )  # Normalize advantages

            # Update actor and critic networks
            self.update(states, actions, old_log_probs, returns, advantages)

            # Log the episode reward
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

            # Save the result to a file every 10 episodes
            if (episode + 1) % 10 == 0:
                with open(save_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([episode + 1, episode_reward])


def main():
    env = create_env()
    sample_observation = env.reset()
    print("Sample observation shape:", sample_observation.shape)
    agent = PPOAgent(env.observation_space.shape, env.action_space.n)

    # Comment out this section If you want to start from scratch
    # ----------------------------------------------------------
    agent.actor.load_state_dict(
        torch.load(ACTOR_PATH, map_location=torch.device("cpu"))
    )
    agent.critic.load_state_dict(
        torch.load(CRITIC_PATH, map_location=torch.device("cpu"))
    )
    # ----------------------------------------------------------

    agent.train(env, 50)
    agent.save("networks")


if __name__ == "__main__":
    main()
