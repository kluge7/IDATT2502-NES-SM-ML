# Hyperparameters for Rainbow DQN Agent
GAMMA = 0.99
BATCH_SIZE = 32
LR = 1e-4
REPLAY_MEMORY_SIZE = 100000
TARGET_UPDATE_FREQUENCY = 1000
MODEL_SAVE_PATH = "ddqn_rainbow.pth"

# Prioritized Experience Replay parameters
ALPHA = 0.6  # How much prioritization is used (0 - no prioritization, 1 - full prioritization)
BETA_START = 0.4  # Initial value of beta for importance-sampling
BETA_FRAMES = (
    100000  # Number of frames over which beta is annealed from initial value to 1
)

# Distributional RL parameters
NUM_ATOMS = 51  # Number of atoms in the distributional output
V_MIN = -10  # Minimum possible value
V_MAX = 10  # Maximum possible value

# Multi-step learning
N_STEPS = 3  # Number of steps for multi-step returns

# NoisyNet parameter
SIGMA_INIT = 0.5  # Initial value of sigma for NoisyLinear layers

# Other parameters remain the same
CSV_FILENAME = "training_results/training_log_rainbow.csv"
RUNS = 50
DELAY = 0.05
MOVING_AVERAGE_WINDOW = 200
NUM_EPISODES = 20000
SAVE_EVERY = 100
