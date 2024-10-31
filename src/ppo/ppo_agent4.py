import csv
import os
import shutil

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from src.environment.environment2 import create_env
from src.ppo.ppo_network import PPO


class PPOAgent:
    def __init__(self, num_states, num_actions, opt):
        self.opt = opt
        self.model = PPO(num_states, num_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)

        # Set up for GPU if available
        if torch.cuda.is_available():
            self.model.cuda()
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        # Prepare paths for saving logs and models
        if os.path.isdir(opt.log_path):
            shutil.rmtree(opt.log_path)
        os.makedirs(opt.log_path)
        if not os.path.isdir(opt.saved_path):
            os.makedirs(opt.saved_path)

    def choose_action(self, state_tensor):
        """Choose an action based on the current state.

        Args:
            state_tensor (torch.Tensor): The current state as a tensor.

        Returns:
            action (torch.Tensor): Chosen action.
            log_prob (torch.Tensor): Log probability of the chosen action.
            value (torch.Tensor): State value estimate.

        """
        # Forward pass through the model
        logits, value = self.model(state_tensor)
        policy = F.softmax(logits, dim=1)
        dist = Categorical(policy)

        # Sample action and calculate log probability
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def train(
        self,
        env,
        num_episodes,
        output_path="training_results",
        output_csv="training_log.csv",
        save_interval=100,
        render=False,
    ):
        """Train the PPO Agent in a single environment, with logging and model saving."""
        os.makedirs(output_path, exist_ok=True)

        # Prepare CSV file for logging training results
        csv_path = os.path.join(output_path, output_csv)
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Got_Flag"])  # Header for CSV file

        # Model saving path
        model_save_path = os.path.join(output_path, "model")

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            got_the_flag = False
            states, actions, old_log_policies, rewards, values, dones = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                if torch.cuda.is_available():
                    state_tensor = state_tensor.cuda()

                # Use choose_action to get action, log_prob, and value
                action, log_prob, value = self.choose_action(state_tensor)

                # Step environment with chosen action
                next_state, reward, done, info = env.step(action.item())
                episode_reward += reward

                # Record experience for updating the policy
                states.append(state_tensor)
                actions.append(action)
                old_log_policies.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)

                # Update state
                state = next_state
                if render:
                    env.render()

                # Check if the flag was obtained
                got_the_flag = info.get("flag_get", False)

            # Process episode rewards and calculate advantages
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(
                0
            )
            if torch.cuda.is_available():
                next_state_tensor = next_state_tensor.cuda()

            with torch.no_grad():
                next_value = self.model(next_state_tensor)[1].squeeze(-1)

            returns, advantages = self.compute_gae(rewards, values, next_value, dones)

            # Prepare tensors
            states = torch.cat(states)
            actions = torch.stack(actions)
            old_log_policies = torch.cat(old_log_policies).detach()

            # PPO Update Step
            self.update(states, actions, old_log_policies, returns, advantages)

            # Log episode results
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

            # Write to CSV
            with open(csv_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([episode + 1, episode_reward, 1 if got_the_flag else 0])

            # Save model periodically
            if (episode + 1) % save_interval == 0:
                model_save_file = f"{model_save_path}_episode_{episode + 1}.pth"
                torch.save(self.model.state_dict(), model_save_file)
                print(f"Model saved at {model_save_file}")

    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation (GAE) for an episode."""
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.opt.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.opt.gamma * self.opt.tau * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        returns = torch.cat(returns).detach()
        advantages = returns - torch.cat(values[:-1]).detach()
        return returns, advantages

    def update(self, states, actions, old_log_policies, returns, advantages):
        """Perform a PPO update with clipped objective and entropy regularization."""
        for _ in range(self.opt.num_epochs):
            indices = torch.randperm(len(states))
            for i in range(0, len(states), self.opt.batch_size):
                batch_indices = indices[i : i + self.opt.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_policies = old_log_policies[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                logits, values = self.model(batch_states)
                new_policy = F.softmax(logits, dim=1)
                dist = Categorical(new_policy)
                new_log_policy = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO loss
                ratio = torch.exp(new_log_policy - batch_old_log_policies)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.opt.epsilon, 1.0 + self.opt.epsilon)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.smooth_l1_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = actor_loss + 0.5 * critic_loss - self.opt.beta * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def eval(self, opt, model, num_states, num_actions):
        pass  # Optional: Implement for evaluation if needed


class Opt:
    """Options for training hyperparameters."""

    lr = 0.001
    gamma = 0.99
    tau = 0.95
    epsilon = 0.1
    beta = 0.01
    num_epochs = 4
    batch_size = 32
    log_path = "logs"
    saved_path = "models"


def main():
    """Sets up the environment and agent, then trains the agent."""
    world, stage, env_version = 1, 1, "v0"
    specification = f"SuperMarioBros-{world}-{stage}-{env_version}"
    env = create_env(map=specification)
    sample_observation = env.reset()
    print("Sample observation shape:", sample_observation.shape)

    # Initialize PPOAgent with observation space and action space
    opt = Opt()
    agent = PPOAgent(sample_observation.shape[0], env.action_space.n, opt)

    agent.model.load_state_dict(
        torch.load(
            "training_results/model_episode_200.pth", map_location=torch.device("cpu")
        )
    )

    # Train the agent with logging, model saving, and optional rendering
    agent.train(
        env,
        num_episodes=1000,
        output_path="training_results",
        output_csv=specification + ".csv",
        render=True,
    )


if __name__ == "__main__":
    main()
