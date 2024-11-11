# Hyperparameters for DDQN Agent
GAMMA = 0.99
BATCH_SIZE = 32
LR = 1e-4
REPLAY_MEMORY_SIZE = 100000
TARGET_UPDATE_FREQUENCY = 1000
EPSILON_START = 0.1
EPSILON_END = 0.1
EPSILON_DECAY = 100000
MODEL_SAVE_PATH = "model/ActionPredictionModel.pth"
CSV_FILENAME = "training_results/training_log_supervised.csv"
# play_trained_agent.py
RUNS = 50
DELAY = 0.05
# plot_training_results.py
MOVING_AVERAGE_WINDOW = 200
# train.py
NUM_EPISODES = 20000
SAVE_EVERY = 100
