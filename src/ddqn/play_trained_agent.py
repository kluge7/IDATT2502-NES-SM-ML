import time

import torch

from src.ddqn.ddqn_agent import DDQNAgent
from src.ddqn.hyperparameters import (
    DDQNHyperparameters,
)  # Import your hyperparameters class
from src.environment.environment import (
    create_env,
)  # Adjust the import path if necessary


def play_trained_agent(hp: DDQNHyperparameters, env):
    in_dim = env.observation_space.shape
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDQNAgent(state_shape=in_dim, action_size=num_actions, device=device, hp=hp)
    agent.load_model(hp.model_save_path)

    for run in range(hp.runs):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        done = False
        total_reward = 0

        print(f"\nRun {run + 1}:")

        while not done:
            env.render()
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
            total_reward += reward

            time.sleep(hp.delay)

        print(f"Total Reward for Run {run + 1}: {total_reward}")

    env.close()


if __name__ == "__main__":
    # Initialize hyperparameters
    hp = DDQNHyperparameters()
    play_trained_agent(hp, create_env())
