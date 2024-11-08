import csv
import os

import hyperparameters as hp
import torch
from ddqn_agent import DQNAgent

from src.environment.environment import create_env


def main():
    num_episodes = hp.NUM_EPISODES
    save_every = hp.SAVE_EVERY

    env = create_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_size = env.action_space.n
    state_shape = env.observation_space.shape
    agent = DQNAgent(state_shape, action_size, device)

    if os.path.exists(hp.MODEL_SAVE_PATH):
        agent.load_model(hp.MODEL_SAVE_PATH)

    with open(hp.CSV_FILENAME, "a", newline="") as csvfile:
        fieldnames = ["Reward", "Got Flag", "Max Distance %"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Only write header if file is new or empty
        if os.path.getsize(hp.CSV_FILENAME) == 0:
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
                    done = True

                agent.memory.push(
                    torch.tensor(state, dtype=torch.float32),
                    action,
                    reward,
                    torch.tensor(next_state, dtype=torch.float32),
                    done,
                )
                state = next_state
                agent.update()

                if agent.steps_done % agent.target_update_frequency == 0:
                    agent.update_target_network()

            max_distance_percentage = (max_x_pos / goal_x_pos) * 100

            if episode % save_every == 0:
                agent.save_model()

            writer.writerow(
                {
                    "Reward": total_reward,
                    "Got Flag": flag_reached,
                    "Max Distance %": max_distance_percentage,
                }
            )
            csvfile.flush()

            # Print statement to monitor progress
            print(
                f"Episode {episode}, Reward: {total_reward}, Got Flag: {flag_reached}, "
                f"Epsilon: {agent.get_epsilon_threshold():.4f}, Max Distance %: {max_distance_percentage:.2f}"
            )

    env.close()


if __name__ == "__main__":
    main()
