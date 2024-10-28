import torch
from ddqn_agent import MODEL_PATH, DDQNAgent

from src.environment.environment import create_env


def play_trained_agent():
    env = create_env()

    # Get input dimensions and number of actions
    in_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # Initialize the agent and load the trained model
    agent = DDQNAgent(state_dim=in_dim, action_dim=num_actions)
    agent.load_model(MODEL_PATH)

    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.select_action(state, epsilon=0.0)  # No exploration during play
        next_state, reward, done, info = env.step(action)
        state = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward

    env.close()
    print(f"Total Reward: {total_reward}")


if __name__ == "__main__":
    play_trained_agent()
