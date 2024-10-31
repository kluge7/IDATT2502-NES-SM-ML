import csv
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from src.environment.environment import create_env
from src.ppo.cnn_network import CNNNetwork
from src.utils import get_unique_filename

ACTOR_PATH = "old_model/actor.pth"
CRITIC_PATH = "old_model/critic.pth"


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent with Advantage Estimation and Actor-Critic Networks.

    This agent implements the PPO algorithm with an actor-critic framework.
    The actor is responsible for action selection based on policy probabilities,
    and the critic estimates the state value for advantage calculation.

    Attributes:
        actor (torch.nn.Module): Neural network representing the policy.
        critic (torch.nn.Module): Neural network representing the value function.
        actor_optimizer (torch.optim.Optimizer): Optimizer for the actor network.
        critic_optimizer (torch.optim.Optimizer): Optimizer for the critic network.
        device (torch.device): Specifies whether to use CPU or GPU for training.

    Hyperparameters:
        gamma (float): Discount factor for future rewards.
        epsilon (float): Clipping parameter for PPO, controlling the policy update constraint.
        lam (float): Lambda parameter for Generalized Advantage Estimation (GAE).
        entropy_coef (float): Coefficient for entropy regularization to encourage exploration.
        grad_clip (float): Maximum gradient norm for gradient clipping.

    """

    def __init__(
        self,
        in_dim,
        num_actions,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        epsilon=0.1,
        lam=0.95,
        entropy_coef=0.01,
        grad_clip=0.5,
    ):
        """Initializes PPOAgent with actor and critic networks, hyperparameters, and optimizers.

        Args:
            in_dim (tuple): Input dimensions for the CNN, representing the state space.
            num_actions (int): Number of actions in the action space.
            actor_lr (float): Learning rate for the actor optimizer.
            critic_lr (float): Learning rate for the critic optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Clipping parameter for PPO to constrain policy updates.
            lam (float): Lambda parameter for GAE (Generalized Advantage Estimation).
            entropy_coef (float): Coefficient for entropy regularization.
            grad_clip (float): Maximum norm for gradient clipping.

        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor and Critic networks
        self.actor = CNNNetwork(in_dim, num_actions).to(self.device)
        self.critic = CNNNetwork(in_dim, 1).to(self.device)

        # Optimizers for Actor and Critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        """Selects an action based on the current policy.

        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            action (int): The chosen action.
            log_prob (torch.Tensor): Log probability of the chosen action.
            value (torch.Tensor): Estimated value of the current state.

        """
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            value = self.critic(state).squeeze(-1)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def compute_gae(self, rewards, values, next_value, dones):
        """Calculates advantages and returns using Generalized Advantage Estimation (GAE).

        Args:
            rewards (list): Collected rewards for each time step in the episode.
            values (list): Estimated state values from the critic.
            next_value (float): Estimated value of the final state.
            dones (list): Boolean indicators for episode completion.

        Returns:
            advantages (torch.Tensor): Computed advantage estimates.
            returns (torch.Tensor): Target returns for each state.

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
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        # Convert lists to tensors after calculating advantages and returns
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(
        self, states, actions, old_log_probs, returns, advantages, mini_batch_size=64
    ):
        """Updates the actor and critic networks using PPO with mini-batch training.

        Args:
            states (torch.Tensor): Batch of observed states.
            actions (torch.Tensor): Batch of actions taken.
            old_log_probs (torch.Tensor): Log probabilities of the actions taken, from the old policy.
            returns (torch.Tensor): Target returns for each state-action pair.
            advantages (torch.Tensor): Computed advantage estimates for each state-action pair.
            mini_batch_size (int): Size of mini-batches for each update epoch.

        """
        for _ in range(4):  # Multiple epochs
            indices = torch.randperm(states.size(0))
            for i in range(0, states.size(0), mini_batch_size):
                batch_indices = indices[i : i + mini_batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Actor update
                logits = self.actor(batch_states)
                probs = F.softmax(logits, dim=-1) + 1e-8  # Adding epsilon to avoid NaNs
                dist = Categorical(probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                    * batch_advantages
                )
                actor_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                )

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.actor_optimizer.step()

                # Critic update
                values = self.critic(batch_states).squeeze(-1)
                critic_loss = F.smooth_l1_loss(values, batch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.critic_optimizer.step()

    def save(self, path="model"):
        """Saves the actor and critic networks to the specified directory.

        Args:
            path (str): Directory to save the model weights.

        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def train(
        self, env, num_episodes, path="training_result", output=None, render=False
    ):
        """Trains the agent in the specified environment using PPO.

        Args:
            env (gym.Env): The environment for training.
            num_episodes (int): Number of episodes to train the agent.
            path (str): Directory to save training logs.
            output (str): Filename for the output CSV file. If None, no CSV file is created.
            render (bool): Whether to render the environment during training.

        """
        os.makedirs(path, exist_ok=True)
        if output:
            output_file = get_unique_filename(path, output)
            with open(os.path.join(path, output_file), mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward", "Got_Flag"])

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
            episode_reward = 0
            got_the_flag = False

            while not done:
                state_tensor = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                action, log_prob, value = self.select_action(state_tensor)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                states.append(state_tensor)
                actions.append(torch.tensor(action, device=self.device))
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)

                state = next_state
                if info.get("flag_get", False):
                    got_the_flag = True

                if render:
                    env.render()

            next_state_tensor = (
                torch.tensor(next_state, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                next_value = self.critic(next_state_tensor).squeeze(-1)

            advantages, returns = self.compute_gae(rewards, values, next_value, dones)
            states = torch.cat(states)
            actions = torch.stack(actions)
            old_log_probs = torch.stack(log_probs)
            returns = returns.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            self.update(states, actions, old_log_probs, returns, advantages)

            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

            if output:
                with open(
                    os.path.join(path, output_file), mode="a", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [episode + 1, episode_reward, 1 if got_the_flag else 0]
                    )

            if episode % 100 == 0:
                self.save("model")

    def train2(
        self,
        env,
        num_episodes,
        output_path="training_result",
        output=None,
        render=False,
    ):
        """Trains the PPO agent in a single environment using a specified number of episodes.

        Args:
            env (gym.Env): The environment for training.
            num_episodes (int): Number of episodes to train the agent.
            output_path (str): Path to save training results.
            output (str): Filename for the output CSV file. If None, no CSV file is created.
            render (bool): Whether to render the environment during training.

        """
        # Set up the training log directory and output file
        os.makedirs(output_path, exist_ok=True)
        if output:
            output_file = get_unique_filename(output_path, output)
            with open(
                os.path.join(output_path, output_file), mode="w", newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward", "Got_Flag"])

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
            episode_reward = 0
            got_the_flag = False

            # Collect experiences for each step in the episode
            while not done:
                state_tensor = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                action, log_prob, value = self.select_action(state_tensor)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # Store the experience
                states.append(state_tensor)
                actions.append(torch.tensor(action, device=self.device))
                log_probs.append(log_prob)
                rewards.append(
                    torch.tensor(reward, dtype=torch.float32, device=self.device)
                )
                values.append(value)
                dones.append(done)

                # Update state and check flag
                state = next_state
                if info.get("flag_get", False):
                    got_the_flag = True

                # Render the environment if specified
                if render:
                    env.render()

            # Compute the next state's value after the episode ends
            next_state_tensor = (
                torch.tensor(next_state, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                next_value = self.critic(next_state_tensor).squeeze(-1)

            # Compute Generalized Advantage Estimation (GAE)
            values.append(next_value)
            gae = 0
            returns = []
            advantages = []

            for step in reversed(range(len(rewards))):
                delta = (
                    rewards[step]
                    + self.gamma * values[step + 1] * (1 - dones[step])
                    - values[step]
                )
                gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[step])

            advantages = torch.tensor(advantages, device=self.device)
            returns = torch.tensor(returns, device=self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Prepare tensors for batch update
            states = torch.cat(states)
            actions = torch.stack(actions)
            old_log_probs = torch.stack(log_probs).detach()

            # PPO Update Step
            for _ in range(4):  # Multiple epochs for each batch
                # Shuffle for mini-batch sampling
                indices = torch.randperm(states.size(0))
                for start in range(
                    0, states.size(0), 64
                ):  # Using 64 as mini-batch size
                    end = start + 64
                    batch_indices = indices[start:end]

                    # Sample batches
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]

                    # Calculate actor (policy) loss
                    logits = self.actor(batch_states)
                    dist = Categorical(F.softmax(logits, dim=-1))
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    # Clipped surrogate objective
                    ratios = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratios * batch_advantages
                    surr2 = (
                        torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                        * batch_advantages
                    )
                    actor_loss = (
                        -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                    )

                    # Update actor network
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.grad_clip
                    )
                    self.actor_optimizer.step()

                    # Calculate critic loss
                    values = self.critic(batch_states).squeeze(-1)
                    critic_loss = F.smooth_l1_loss(values, batch_returns)

                    # Update critic network
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.grad_clip
                    )
                    self.critic_optimizer.step()

            # Log episode results
            print(
                f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Total Loss: {actor_loss + critic_loss}"
            )

            if output:
                with open(
                    os.path.join(output_path, output_file), mode="a", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [episode + 1, episode_reward, 1 if got_the_flag else 0]
                    )

            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save("model")


def main():
    """Sets up the environment and agent, then trains the agent."""
    world, stage, env_version = 1, 1, "v0"
    specification = f"SuperMarioBros-{world}-{stage}-{env_version}"
    env = create_env(map=specification)
    sample_observation = env.reset()
    print("Sample observation shape:", sample_observation.shape)
    agent = PPOAgent(env.observation_space.shape, env.action_space.n)

    # agent.actor.load_state_dict(
    #     torch.load(ACTOR_PATH, map_location=torch.device("cpu"))
    # )
    # agent.critic.load_state_dict(
    #     torch.load(CRITIC_PATH, map_location=torch.device("cpu"))
    # )

    agent.train2(env, 1000, render=True, output=specification + ".csv")
    agent.save()


if __name__ == "__main__":
    main()
