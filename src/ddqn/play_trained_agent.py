import time

import hyperparameters as hp
import torch
from ddqn_agent import DDQNAgent

from src.environment.environment import create_env


def play_trained_agent():
    env = create_env()

    in_dim = env.observation_space.shape
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDQNAgent(state_shape=in_dim, action_size=num_actions, device=device)
    agent.load_model(hp.MODEL_SAVE_PATH)

    for run in range(hp.RUNS):
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

            time.sleep(hp.DELAY)

        print(f"Total Reward for Run {run + 1}: {total_reward}")

    env.close()


if __name__ == "__main__":
    play_trained_agent()
