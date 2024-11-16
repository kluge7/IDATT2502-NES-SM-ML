import csv
import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from torch import nn, optim
from torch.distributions import Categorical

from src.environment.environment import create_env
from src.network.cnn_network import CNNNetwork
from src.ppo.ppo_hyperparameters import PPOHyperparameters
from src.utils import get_unique_filename


class PPOAgent:
    def __init__(self, env, options: PPOHyperparameters):
        """Initializes the PPO agent with actor and critic networks and hyperparameters.

        Args:
            env (gym.Env): The environment in which the agent will operate.
            options (PPOHyperparameters): Contains hyperparameters for training and evaluation.

        Attributes:
            actor (CNNNetwork): The neural network representing the policy (actor).
            critic (CNNNetwork): The neural network representing the value function (critic).
            actor_optimizer (torch.optim.Optimizer): Optimizer for the actor network.
            critic_optimizer (torch.optim.Optimizer): Optimizer for the critic network.
            device (torch.device): Device to run computations on (GPU if available, else CPU).
        """
        self.env = env
        self.options = options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dimension = env.observation_space.shape
        self.action_dimension = env.action_space.n

        self.actor = CNNNetwork(self.obs_dimension[0], self.action_dimension).to(
            self.device
        )
        self.critic = CNNNetwork(self.obs_dimension[0], 1).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=options.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=options.lr)

    def select_action(self, observation):
        """Selects an action and log probability based on the current policy.

        Args:
            observation (np.array): The current state observation from the environment.

        Returns:
            selected_action (int): The chosen action index.
            log_probability (torch.Tensor): The log probability of the chosen action.
        """
        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        action_logits = self.actor(observation.unsqueeze(0).to(self.device))
        action_probabilities = F.softmax(action_logits, dim=-1)[0]
        action_distribution = Categorical(action_probabilities)

        # Sample an action and get the log probability
        selected_action = action_distribution.sample()
        log_probability = action_distribution.log_prob(selected_action)
        return selected_action.item(), log_probability.detach().to(self.device)

    def calculate_gae(self, episode_rewards, episode_values, episode_dones):
        """Calculates the Generalized Advantage Estimation (GAE) for each step.

        Recursive formula: Aₜ = δₜ + γ * λ * (1 - doneₜ) * Aₜ₊₁

        Args:
            episode_rewards (list of list): List of rewards for each episode.
            episode_values (list of list): List of value estimates from the critic for each episode.
            episode_dones (list of list): List of booleans indicating episode termination.

        Returns:
            batch_advantages (torch.Tensor): Calculated advantages for each step in the batch.
        """
        batch_advantages = []

        for rewards, values, dones in zip(
            episode_rewards, episode_values, episode_dones
        ):
            episode_advantages = []
            last_advantage = 0

            for step in reversed(range(len(rewards))):
                if step + 1 < len(rewards):
                    # The temporal difference error δₜ (delta)
                    # δₜ = rₜ + γ * V(sₜ₊₁) * (1 - doneₜ) - V(sₜ)
                    delta = (
                        rewards[step]
                        + self.options.gamma * values[step + 1] * (1 - dones[step])
                        - values[step]
                    )
                else:
                    delta = rewards[step] - values[step]

                # The Aₜ (Generalized Advantage Estimation (GAE))
                # Aₜ = δₜ + γ * λ * (1 - doneₜ) * Aₜ₊₁
                advantage = (
                    delta
                    + self.options.gamma
                    * self.options.lam
                    * (1 - dones[step])
                    * last_advantage
                )

                last_advantage = advantage
                episode_advantages.insert(0, advantage)

            batch_advantages.extend(episode_advantages)

        return torch.tensor(batch_advantages, dtype=torch.float32).to(self.device)

    def evaluate(self, batch_observations, batch_actions):
        r"""Evaluates value estimates V(s) and action log probabilities for a batch.

        Given a state s, the value function V(s) is defined as:
        V(s) = E[Σₖ₌₀ⁿ (γᵏ * rₜ₊ₖ | sₜ = s)]
        where:
        - V(s) is the expected return from state s
        - γ is the discount factor (0 < γ < 1)
        - rₜ₊ₖ is the reward received k steps into the future
        - The critic network approximates V(s) based on this expectation.

        Args:
            batch_observations (torch.Tensor): Batch of observations.
            batch_actions (torch.Tensor): Batch of actions taken.

        Returns:
            v (torch.Tensor): Value estimates for each observation.
            log_action_probabilities (torch.Tensor): Log probabilities of actions.
            entropy (torch.Tensor): Entropy of the action distribution for exploration regularization.
        """
        # Approximate V(s)
        # v[i] = V(s[i]) ≈ critic_network(s[i])
        v = self.critic(batch_observations).squeeze()

        batch_observations = torch.tensor(batch_observations, dtype=torch.float).to(
            self.device
        )
        action_logits = self.actor(batch_observations)
        action_probabilities = F.softmax(action_logits, dim=-1).to(self.device)
        action_distribution = Categorical(action_probabilities)
        log_action_probabilities = action_distribution.log_prob(batch_actions).to(
            self.device
        )

        return v, log_action_probabilities, action_distribution.entropy()

    def rollout(self, episode_output_file):
        """Collects data for a batch by running episodes in the environment.

        Args:
            episode_output_file (str): Path to save episode results in CSV format.

        Returns:
            tuple: Contains observations, actions, log probabilities, rewards, lengths, values,
                   dones, and flag count for the batch.
        """
        gc.collect()

        batch_observations = []
        batch_actions = []
        batch_log_probabilities = []
        batch_rewards = []
        batch_lengths = []
        batch_values = []
        batch_dones = []
        batch_flags = 0

        timestep = 0

        while timestep < self.options.timesteps_per_batch:
            episode_rewards = []
            episode_values = []
            episode_dones = []

            observations = self.env.reset()
            done = False
            flag = False

            def collect_data(episode_rewards, episode_values, episode_dones):
                """Collects observations, actions, values, rewards, and dones for a batch."""
                nonlocal observations, done, flag, timestep

                if self.options.render:
                    self.env.render()

                timestep += 1
                episode_dones.append(done)
                batch_observations.append(observations)

                action, log_probabilities = self.select_action(observations)
                v = self.critic(
                    torch.tensor(observations, dtype=torch.float)
                    .unsqueeze(0)
                    .to(self.device)
                )
                observations, reward, done, info = self.env.step(action)
                if info.get("flag_get"):
                    flag = True

                episode_rewards.append(reward)
                episode_values.append(v.flatten())
                batch_actions.append(action)
                batch_log_probabilities.append(log_probabilities)

            # Run the main episode loop
            for _episode_step in range(self.options.max_timesteps_per_episode):
                collect_data(episode_rewards, episode_values, episode_dones)
                if done:
                    break

            # If we are close to the timestep limit, continue until the episode finishes
            if (
                timestep + self.options.max_timesteps_per_episode
                >= self.options.timesteps_per_batch
            ) and not done:
                while not done:
                    collect_data(episode_rewards, episode_values, episode_dones)

            batch_lengths.append(_episode_step + 1)
            batch_rewards.append(episode_rewards)
            batch_values.append(episode_values)
            batch_dones.append(episode_dones)

            with open(
                episode_output_file,
                mode="a",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow([np.sum(episode_rewards), 1 if flag else 0])

            # tensorboard_writer.add_scalar("Training/Episode/Reward", np.sum(episode_rewards))

            if flag:
                batch_flags += 1

        batch_observations = torch.tensor(
            np.array(batch_observations), dtype=torch.float32
        ).to(self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(self.device)
        batch_log_probabilities = torch.tensor(
            batch_log_probabilities, dtype=torch.float32
        ).to(self.device)

        return (
            batch_observations,
            batch_actions,
            batch_log_probabilities,
            batch_rewards,
            batch_lengths,
            batch_values,
            batch_dones,
            batch_flags,
        )

    def update(
        self,
        batch_observations,
        batch_actions,
        batch_log_probabilities,
        batch_rtgs,
        advantages,
        timestep,
        max_timesteps,
    ):
        r"""Updates the actor and critic networks using PPO by using clipped surrogate objective for the actor and Mean Squared Error (MSE) for the critic.

        PPO policy update:
        Lₜ_clip = E[min(rₜ(θ) * Aₜ, clip(rₜ(θ), 1 - ε, 1 + ε) * Aₜ)]
        where:
        - rₜ(θ) is the policy ratio: rₜ(θ) = π_θ(aₜ | sₜ) / π_θ₋₁(aₜ | sₜ)
        - Aₜ is the advantage estimate for each state-action pair
        - ε is a clipping parameter to prevent large policy updates.

        Critic value update:
        critic_loss = MSE(v, Rₜ)
        where:
        - v is the predicted value for each state, V(s)
        - Rₜ is the target return for each state-action pair.

        Args:
            batch_observations (torch.Tensor): Batch of observations.
            batch_actions (torch.Tensor): Batch of actions taken.
            batch_log_probabilities (torch.Tensor): Log probabilities from the previous policy.
            batch_rtgs (torch.Tensor): Target returns for each state-action pair.
            advantages (torch.Tensor): Computed advantages for each state-action pair.
            timestep (int): Current timestep in the training loop.
            max_timesteps (int): Maximum number of timesteps to train.

        Returns:
            float: Average actor loss and critic loss for the update.
        """
        step = batch_observations.size(0)
        indices = np.arange(step)
        minibatch_size = step // self.options.num_minibatches
        actor_losses = []
        critic_losses = []

        for _ in range(self.options.n_updates_per_iteration):
            if self.options.dynamic_lr:
                frac = (timestep - 1.0) / max_timesteps
                new_lr = self.options.lr * (1.0 - frac)
                new_lr = max(new_lr, self.options.min_lr_limit)
                self.actor_optimizer.param_groups[0]["lr"] = new_lr
                self.critic_optimizer.param_groups[0]["lr"] = new_lr

            np.random.shuffle(indices)

            for start in range(0, step, minibatch_size):
                end = start + minibatch_size
                idx = indices[start:end]

                mini_observations = batch_observations[idx]
                mini_actions = batch_actions[idx]
                mini_log_probabilities = batch_log_probabilities[idx]
                mini_advantages = advantages[idx]
                mini_rtgs = batch_rtgs[idx]

                v, current_log_probabilities, entropy = self.evaluate(
                    mini_observations, mini_actions
                )

                # Policy ratio: rₜ(θ) = exp(log_pi - log_pi_old)
                ratios = torch.exp(current_log_probabilities - mini_log_probabilities)

                # KL (Kullback-Leibler divergence) measures the difference between the old and new policies
                # KL = E[log(π_old(aₜ | sₜ) / π_new(aₜ | sₜ))]
                kl = (mini_log_probabilities - current_log_probabilities).mean()
                # Alternatively:
                # kl = (
                #     (ratios - 1) ** (current_log_probabilities - mini_log_probabilities)
                # ).mean()

                # Clipped Surrogate Objective terms:
                # surr1 = rₜ(θ) * Aₜ
                surr1 = ratios * mini_advantages

                # surr2 = clip(rₜ(θ), 1 - ε, 1 + ε) * Aₜ
                surr2 = (
                    torch.clamp(ratios, 1 - self.options.clip, 1 + self.options.clip)
                    * mini_advantages
                )

                # Actor loss with entropy regularization:
                # actor_loss = -mean(min(surr1, surr2)) - entropy_coef * entropy
                actor_loss = (
                    -torch.min(surr1, surr2)
                ).mean() - self.options.ent_coef * entropy.mean()

                # Critic loss (Mean Squared Error):
                # critic_loss = MSE(v, Rₜ)
                critic_loss = nn.MSELoss()(v, mini_rtgs)

                # Perform backpropagation and optimization step for the actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.options.max_grad_norm
                )
                self.actor_optimizer.step()

                # Perform backpropagation and optimization step for the critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.options.max_grad_norm
                )
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.detach())
                critic_losses.append(critic_loss.detach())

            if self.options.kl_divergence and kl > self.options.target_kl:
                break

        return np.mean(actor_losses), np.mean(critic_losses)

    def train(self, max_timesteps):
        """Trains the agent in the environment using PPO.

        Args:
            max_timesteps (int): Total number of timesteps for training.
        """
        # tensorboard_writer = SummaryWriter(log_dir=self.options.tensorboard_log_dir)

        os.makedirs(self.options.model_path, exist_ok=True)
        os.makedirs(self.options.training_result_path, exist_ok=True)
        os.makedirs(self.options.episode_result_path, exist_ok=True)

        output_csv = self.options.specification + ".csv"
        training_output_file = get_unique_filename(
            self.options.training_result_path, output_csv
        )
        episode_output_file = get_unique_filename(
            self.options.episode_result_path, output_csv
        )

        with open(
            training_output_file,
            mode="w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Iteration",
                    "BatchReward",
                    "GotFlag",
                    "AverageActorLoss",
                    "PercentageCompleted",
                ]
            )

        with open(
            episode_output_file,
            mode="w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Reward", "GotFlag"])

        timestep, iteration = 0, 0

        while timestep < max_timesteps:
            (
                batch_observations,
                batch_actions,
                batch_log_probabilities,
                batch_rewards,
                batch_lengths,
                batch_values,
                batch_dones,
                batch_flags,
            ) = self.rollout(episode_output_file)

            timestep += np.sum(batch_lengths)
            iteration += 1

            # Advantages for each time step using Generalized Advantage Estimation (GAE)
            advantages = self.calculate_gae(batch_rewards, batch_values, batch_dones)
            # Value estimates for each observation in the batch using the critic network
            v = self.critic(batch_observations).squeeze()
            # Target returns for the critic by adding advantages to the detached value estimates
            batch_rtgs = advantages + v.detach()
            # Normalize advantages to stabilize training updates
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            average_actor_loss, average_critic_loss = self.update(
                batch_observations,
                batch_actions,
                batch_log_probabilities,
                batch_rtgs,
                advantages,
                timestep,
                max_timesteps,
            )

            # # Log to TensorBoard
            # tensorboard_writer.add_scalar("Training/Epoch/AverageActorLoss", average_actor_loss, iteration)
            # tensorboard_writer.add_scalar("Training/Epoch/AverageCriticLoss", average_critic_loss, iteration)
            # tensorboard_writer.add_scalar("Training/Epoch/Reward", np.sum(np.concatenate(batch_rewards)), iteration)
            # tensorboard_writer.add_scalar("Training/Epoch/GotFlag", batch_flags, iteration)

            with open(
                training_output_file,
                mode="a",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        iteration,
                        np.sum(np.concatenate(batch_rewards)),
                        batch_flags,
                        average_actor_loss,
                        f"{timestep * 100 / max_timesteps:.2f}%",
                    ]
                )

            print(
                f"Iteration: {iteration}, Batch rewards: {np.sum(np.concatenate(batch_rewards))}, Got flag: {batch_flags}, Average actor loss = {average_actor_loss:.2f}, Percentage completed: {timestep * 100 / max_timesteps:.2f}%"
            )

            if iteration % self.options.save_freq == 0:
                self.save_networks()

        # tensorboard_writer.close()

    def save_networks(self):
        """Saves the actor and critic network weights to the specified paths."""
        torch.save(
            self.actor.state_dict(),
            str(os.path.join(self.options.model_path, self.options.model_actor)),
        )
        torch.save(
            self.critic.state_dict(),
            str(os.path.join(self.options.model_path, self.options.model_critic)),
        )

    def load_networks(self, actor_path=None, critic_path=None):
        """Loads the actor and critic network weights from the specified paths."""
        try:
            if actor_path is None:
                actor_path = os.path.join(
                    self.options.model_path, self.options.model_actor
                )
            if critic_path is None:
                critic_path = os.path.join(
                    self.options.model_path, self.options.model_critic
                )

            # Use map_location to load model to CPU if GPU is not available
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )
        except FileNotFoundError:
            print(
                f"Error: Could not find model files at {actor_path} or {critic_path}. Starting training from scratch..."
            )

    def test(self, render=True, episodes=1):
        """Plays episodes using the trained actor network without updating it.

        Args:
            render (bool): If True, renders the environment during the episode.
            episodes (int): Number of episodes to play.
        """
        success_rate_list = []

        for _ in range(episodes):
            observation = self.env.reset()
            done = False
            total_reward = 0
            step_count = 0
            flag_reached = False  # Track if the agent reached the flag

            while not done:
                if render:
                    self.env.render()

                # Select action based on the current policy (actor network)
                action, _ = self.select_action(observation)

                # Take the selected action in the environment
                observation, reward, done, info = self.env.step(action)

                # Check if the flag was reached
                if info.get("flag_get"):
                    flag_reached = True
                    success_rate_list.append(1)

                # Accumulate the reward for the episode
                total_reward += reward
                step_count += 1

            print(
                f"Episode finished in {step_count} steps with total reward: {total_reward}. Flag reached: {'Yes' if flag_reached else 'No'}"
            )
        print(f"Success rate: {(np.sum(success_rate_list) / episodes)*100:.2f}%")


def main():
    """Sets up the environment and agent, then trains the agent.

    Environment:
        Super Mario Bros environment is created using the `create_env` function.

    Agent:
        PPO agent is created and trained for a specified number of timesteps.
    """
    world, stage, env_version = 1, 1, "v0"
    specification = f"SuperMarioBros-{world}-{stage}-{env_version}"
    env = create_env(map=specification, skip=4, actions=COMPLEX_MOVEMENT)

    options = PPOHyperparameters(render=True, specification=specification)

    agent = PPOAgent(env, options)
    agent.load_networks(
        actor_path="model/ppo_actor_no_supervised.pth",
        critic_path="model/ppo_critic_no_supervised.pth",
    )

    total_timesteps = 3_000_000
    agent.train(max_timesteps=total_timesteps)


if __name__ == "__main__":
    main()
