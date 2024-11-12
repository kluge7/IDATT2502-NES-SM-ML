# Hyperparameters for DQN Agent
GAMMA = 0.99
BATCH_SIZE = 32
LR = 1e-4
REPLAY_MEMORY_SIZE = 100000
TARGET_UPDATE_FREQUENCY = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 100000
MODEL_SAVE_PATH = "model/dqn_1-1.pth"
CSV_FILENAME = "training_results/training_log_1-1.csv"

# Prioritized Replay Buffer parameters
PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_FRAMES = 100000

# play_trained_agent.py
RUNS = 50
DELAY = 0.05

# plot_training_results.py
MOVING_AVERAGE_WINDOW = 250

# train.py
NUM_EPISODES = 20000
SAVE_EVERY = 100
