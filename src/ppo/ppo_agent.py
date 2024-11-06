import csv
import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from cnn_network import CNNNetwork
from ppo_hyperparameters import PPOHyperparameters
from torch import nn, optim
from torch.distributions import Categorical

from src.environment.environment import create_env
from src.utils import get_unique_filename


class PPOAgent:
    def __init__(self, env, options: PPOHyperparameters):
        self.env = env
        self.options = options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dimension = env.observation_space.shape
        self.action_dimension = env.action_space.n

        self.actor = CNNNetwork(self.obs_dimension, self.action_dimension).to(
            self.device
        )
        self.critic = CNNNetwork(self.obs_dimension, 1).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=options.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=options.lr)

    def select_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        action_logits = self.actor(observation.unsqueeze(0).to(self.device))
        action_probabilities = F.softmax(action_logits, dim=-1)[0]
        action_distribution = Categorical(action_probabilities)

        # Sample an action and get the log probability
        selected_action = action_distribution.sample()
        log_probability = action_distribution.log_prob(selected_action)
        return selected_action.item(), log_probability.detach().to(self.device)

    def calculate_gae(self, episode_rewards, episode_values, episode_dones):
        batch_advantages = []

        for rewards, values, dones in zip(
            episode_rewards, episode_values, episode_dones
        ):
            episode_advantages = []
            last_advantage = 0

            for step in reversed(range(len(rewards))):
                if step + 1 < len(rewards):
                    delta = (
                        rewards[step]
                        + self.options.gamma * values[step + 1] * (1 - dones[step])
                        - values[step]
                    )
                else:
                    delta = rewards[step] - values[step]

                # Calculate the GAE advantage
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

            for _episode_step in range(self.options.max_timesteps_per_episode):
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

                if done:
                    break

            batch_lengths.append(_episode_step + 1)
            batch_rewards.append(episode_rewards)
            batch_values.append(episode_values)
            batch_dones.append(episode_dones)

            with open(
                os.path.join(self.options.episode_result_path, episode_output_file),
                mode="a",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow([np.sum(episode_rewards), 1 if flag else 0])

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
        step = batch_observations.size(0)
        indices = np.arange(step)
        minibatch_size = step // self.options.num_minibatches
        loss = []

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

                ratios = torch.exp(current_log_probabilities - mini_log_probabilities)
                kl = (
                    (ratios - 1) ** (current_log_probabilities - mini_log_probabilities)
                ).mean()

                surr1 = ratios * mini_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.options.clip, 1 + self.options.clip)
                    * mini_advantages
                )
                actor_loss = (
                    -torch.min(surr1, surr2)
                ).mean() - self.options.ent_coef * entropy.mean()
                critic_loss = nn.MSELoss()(v, mini_rtgs)

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.options.max_grad_norm
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.options.max_grad_norm
                )
                self.critic_optimizer.step()

                loss.append(actor_loss.detach())

            if kl > self.options.target_kl:
                break

        return np.mean(loss)

    def train(
        self, max_timesteps, model_actor="ppo_actor.pth", model_critic="ppo_critic.pth"
    ):
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
            os.path.join(self.options.training_result_path, training_output_file),
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
            os.path.join(self.options.episode_result_path, episode_output_file),
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

            advantages = self.calculate_gae(batch_rewards, batch_values, batch_dones)
            v = self.critic(batch_observations).squeeze()
            batch_rtgs = advantages + v.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            loss = self.update(
                batch_observations,
                batch_actions,
                batch_log_probabilities,
                batch_rtgs,
                advantages,
                timestep,
                max_timesteps,
            )

            with open(
                os.path.join(self.options.training_result_path, training_output_file),
                mode="a",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        iteration,
                        np.sum(np.concatenate(batch_rewards)),
                        batch_flags,
                        loss,
                        f"{timestep * 100 / max_timesteps:.2f}%",
                    ]
                )

            print(
                f"Iteration: {iteration}, Batch rewards: {np.sum(np.concatenate(batch_rewards))}, Got flag: {batch_flags}, Average actor loss = {loss:.2f}, Percentage completed: {timestep * 100 / max_timesteps:.2f}%"
            )

            if iteration % self.options.save_freq == 0:
                torch.save(
                    self.actor.state_dict(),
                    os.path.join(self.options.model_path, model_actor),
                )
                torch.save(
                    self.critic.state_dict(),
                    os.path.join(self.options.model_path, model_critic),
                )


def main():
    world, stage, env_version = 1, 1, "v0"
    specification = f"SuperMarioBros-{world}-{stage}-{env_version}"
    env = create_env(map=specification, action_repeat=4)
    options = PPOHyperparameters(render=True, specification=specification)
    agent = PPOAgent(env, options)

    # agent.actor.load_state_dict(
    #     torch.load("model/ppo_actor.pth", map_location=torch.device("cpu"))
    # )
    # agent.critic.load_state_dict(
    #     torch.load("model/ppo_critic.pth", map_location=torch.device("cpu"))
    # )

    total_timesteps = 3_000_000
    agent.train(max_timesteps=total_timesteps)


if __name__ == "__main__":
    main()
