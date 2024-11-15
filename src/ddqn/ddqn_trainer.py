import csv
import os

import torch

from src.ddqn.ddqn_agent import DDQNAgent
from src.ddqn.hyperparameters import DDQNHyperparameters
from src.environment.environment import create_env


def train(hp: DDQNHyperparameters):
    num_episodes = hp.num_episodes
    save_every = hp.save_every

    env = create_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))

    action_size = env.action_space.n
    state_shape = env.observation_space.shape
    agent = DDQNAgent(state_shape, action_size, device, hp)

    if os.path.exists(hp.model_save_path):
        agent.load_model(hp.model_save_path)

    with open(hp.csv_filename, "a", newline="") as csvfile:
        fieldnames = ["Reward", "Got Flag", "Max Distance %"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file is empty
        if os.path.getsize(hp.csv_filename) == 0:
            writer.writeheader()

        for episode in range(1, num_episodes + 1):
            state = env.reset()
            total_reward = 0
            done = False
            flag_reached = False
            max_x_pos = 0
            goal_x_pos = 3300

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                max_x_pos = max(max_x_pos, info.get("x_pos", 0))
                if info.get("flag_get", False):
                    flag_reached = True

                agent.memory.push(
                    torch.tensor(state, dtype=torch.float32),
                    action,
                    reward,
                    torch.tensor(next_state, dtype=torch.float32),
                    done,
                )
                state = next_state
                agent.update()

                if agent.steps_done % hp.target_update_frequency == 0:
                    agent.update_target_network()

            max_distance_percentage = (max_x_pos / goal_x_pos) * 100

            if episode % save_every == 0:
                agent.save_model(hp.model_save_path)

            writer.writerow(
                {
                    "Reward": total_reward,
                    "Got Flag": flag_reached,
                    "Max Distance %": max_distance_percentage,
                }
            )
            csvfile.flush()

            print(
                f"Episode {episode}, Reward: {total_reward}, Got Flag: {flag_reached}, "
                f"Epsilon: {agent.get_epsilon_threshold():.4f}, Max Distance %: {max_distance_percentage:.2f}"
            )

    env.close()


if __name__ == "__main__":
    hp = DDQNHyperparameters()
    train(hp)
