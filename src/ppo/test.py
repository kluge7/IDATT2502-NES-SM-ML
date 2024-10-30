import torch

from src.environment.environment import (
    create_env,
)
from src.ppo.ppo_agent import PPOAgent

# Paths to the saved model files
ACTOR_PATH = "model/actor.pth"
CRITIC_PATH = "model/critic.pth"

# Set device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_agent(in_dim, num_actions, actor_path, critic_path):
    """Initialize a PPOAgent and load the trained actor and critic model.

    Args:
        in_dim (tuple): Input dimensions (shape) for the CNN.
        num_actions (int): Number of actions in the action space.
        actor_path (str): Path to the saved actor model.
        critic_path (str): Path to the saved critic model.

    Returns:
        PPOAgent: The PPO agent with loaded weights on the appropriate device (GPU if available).

    """
    agent = PPOAgent(in_dim, num_actions)

    # Load the trained model weights to the device
    agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    agent.critic.load_state_dict(torch.load(critic_path, map_location=device))

    # Move models to the device
    agent.actor.to(device)
    agent.critic.to(device)

    agent.actor.eval()
    agent.critic.eval()

    return agent


def play_single_game(agent, env):
    """Plays a single game using the provided agent and environment."""
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action, _, _ = agent.select_action(state_tensor)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        state = next_state

    env.close()
    print("Total Reward:", total_reward)


def main():
    """Main function to initialize the environment, load the agent, and play a single game."""
    env = create_env(
        map="SuperMarioBros-1-1-v0", output_path="output/SuperMarioBros-1-1-v0.mp4"
    )

    in_dim = env.observation_space.shape
    num_actions = env.action_space.n
    agent = load_trained_agent(in_dim, num_actions, ACTOR_PATH, CRITIC_PATH)

    play_single_game(agent, env)


if __name__ == "__main__":
    main()
