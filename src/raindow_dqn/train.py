import csv
import os

import hyperparameters as hp
import torch
from rainbow_agent import RainbowAgent

from src.environment.environment import create_env


def main():
    num_episodes = hp.NUM_EPISODES

    # Create the environment and initialize device
    env = create_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_size = env.action_space.n
    state_shape = env.observation_space.shape
    agent = RainbowAgent(state_shape, action_size, device)

    # Load pre-trained model if it exists
    if os.path.exists(hp.MODEL_SAVE_PATH):
        agent.load_model(hp.MODEL_SAVE_PATH)

    # Set up CSV logging
    with open(hp.CSV_FILENAME, "a", newline="") as csvfile:
        fieldnames = ["Reward", "Got Flag", "Max Distance %"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Only write header if file is new or empty
        if os.path.getsize(hp.CSV_FILENAME) == 0:
            writer.writeheader()

        # Training loop
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

                # Push transition to replay memory
                agent.memory.push(
                    torch.tensor(state, dtype=torch.float32),
                    action,
                    reward,
                    torch.tensor(next_state, dtype=torch.float32),
                    done,
                )
                state = next_state

                # Perform a training step
                agent.update()

                # Update target network periodically
                if agent.steps_done % agent.target_update_frequency == 0:
                    agent.update_target_network()

            # Calculate max distance percentage
            max_distance_percentage = (max_x_pos / goal_x_pos) * 100

            # Save model every 500 episodes with unique filename
            if episode % 500 == 0:
                model_path = f"model_checkpoints/rainbow_model_ep{episode}.pth"
                os.makedirs("model_checkpoints", exist_ok=True)
                agent.save_model(model_path)
                print(f"Saved model at {model_path}")

            # Save the latest model every 100 episodes
            if episode % 100 == 0:
                latest_model_path = "model_checkpoints/latest_rainbow_model.pth"
                agent.save_model(latest_model_path)
                print(f"Updated latest model at {latest_model_path}")

            # Log metrics to CSV
            writer.writerow(
                {
                    "Reward": total_reward,
                    "Got Flag": flag_reached,
                    "Max Distance %": max_distance_percentage,
                }
            )
            csvfile.flush()

            # Print progress
            print(
                f"Episode {episode}, Reward: {total_reward}, Got Flag: {flag_reached}, "
                f"Max Distance %: {max_distance_percentage:.2f}, "
            )

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
