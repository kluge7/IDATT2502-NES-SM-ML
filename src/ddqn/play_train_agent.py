import torch
from src.environment.environment import create_env
from ddqn_agent import DQNAgent  # Assuming ddqn_agent.py contains your DQNAgent class

MODEL_PATH = 'dqn_model_new.pth'  # Make sure this matches the save path in DQNAgent

# Define the play_trained_agent function to run and render the trained model
def play_trained_agent(runs=50):
    env = create_env()

    # Get input dimensions and number of actions
    in_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # Initialize the agent, ensure device compatibility, and load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_shape=in_dim, action_size=num_actions, device=device)
    agent.load_model(MODEL_PATH)  # Use the model path for loading

    # Run the agent in the environment for the specified number of runs
    for run in range(runs):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        done = False
        total_reward = 0

        print(f"\nRun {run + 1}:")

        # Run through the environment until done
        while not done:
            env.render()
            action = agent.select_action(state)  # No exploration during play
            next_state, reward, done, info = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
            total_reward += reward

        print(f"Total Reward for Run {run + 1}: {total_reward}")

    env.close()

if __name__ == "__main__":
    play_trained_agent()
