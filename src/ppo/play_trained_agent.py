import torch

from src.environment.environment2 import (
    create_env,
)  # Import the environment setup function
from src.ppo.ppo_agent import PPOAgent  # Import the PPOAgent class

# Paths to the saved model files
ACTOR_PATH = "networks/actor.pth"
CRITIC_PATH = "networks/critic.pth"


def load_trained_agent(in_dim, num_actions, actor_path, critic_path):
    """Initialize a PPOAgent and load the trained actor and critic networks.

    Args:
        in_dim (tuple): Input dimensions (shape) for the CNN.
        num_actions (int): Number of actions in the action space.
        actor_path (str): Path to the saved actor model.
        critic_path (str): Path to the saved critic model.

    Returns:
        PPOAgent: The PPO agent with loaded weights.

    """
    # Initialize the PPO agent
    agent = PPOAgent(in_dim, num_actions)

    # Load the trained model weights
    agent.actor.load_state_dict(
        torch.load(actor_path, map_location=torch.device("cpu"))
    )
    agent.critic.load_state_dict(
        torch.load(critic_path, map_location=torch.device("cpu"))
    )

    # Set models to evaluation mode
    agent.actor.eval()
    agent.critic.eval()

    return agent


def play_single_game(agent, env):
    """Plays a single game using the provided agent and environment.

    Args:
        agent (PPOAgent): The trained PPO agent.
        env: The Super Mario Bros environment.

    """
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Convert state to tensor and add batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Select an action using the trained agent
        action, _, _ = agent.select_action(state_tensor)

        # Step the environment with the selected action
        next_state, reward, done, info = env.step(action)

        # Accumulate rewards for logging
        total_reward += reward

        # Render the environment to visualize the agent playing
        env.render()

        # Update state for the next step
        state = next_state

    env.close()  # Close the environment window
    print("Total Reward:", total_reward)


def main():
    # Create the Super Mario Bros environment
    env = create_env()

    # Initialize the input dimensions and number of actions from the environment
    in_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # Load the trained PPO agent
    agent = load_trained_agent(in_dim, num_actions, ACTOR_PATH, CRITIC_PATH)

    # Play a single game with the loaded agent
    play_single_game(agent, env)


if __name__ == "__main__":
    main()
