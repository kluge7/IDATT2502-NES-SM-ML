class DDQNHyperparameters:
    """Defines hyperparameters for training a Double Deep Q-Network (DDQN) agent.

    Attributes:
        gamma (float): Discount factor for future rewards, controlling the importance of long-term vs. immediate rewards.
        batch_size (int): Number of experiences sampled from the replay memory in each update.
        lr (float): Learning rate for the optimizer.
        replay_memory_size (int): Capacity of the replay memory buffer for storing past experiences.
        target_update_frequency (int): Number of steps between updates of the target network.

        epsilon_start (float): Initial value of epsilon for epsilon-greedy exploration.
        epsilon_end (float): Final value of epsilon for epsilon-greedy exploration.
        epsilon_decay (int): Number of frames over which epsilon is linearly decayed.

        model_save_path (str): Path to save the trained DQN model checkpoint.
        csv_filename (str): Filename for saving training logs in CSV format.

        priority_alpha (float): Alpha parameter for prioritized experience replay, controls how much prioritization is used.
        priority_beta_start (float): Initial value of beta for importance-sampling weights in prioritized replay.
        priority_beta_frames (int): Number of frames over which beta is linearly increased to 1.0.

        runs (int): Number of times to run a trained agent for evaluation or demo purposes.
        delay (float): Delay between actions in seconds for running a trained agent for demonstration.

        moving_average_window (int): Window size for calculating the moving average when plotting training results.

        num_episodes (int): Total number of episodes to train the DQN agent.
        save_every (int): Frequency (in episodes) for saving the model during training.
    """

    def __init__(
        self,
        # Main DQN hyperparameters
        gamma=0.99,
        batch_size=32,
        lr=1e-4,
        replay_memory_size=100000,
        target_update_frequency=1000,
        # Exploration parameters
        epsilon_start=0.01,
        epsilon_end=0.01,
        epsilon_decay=100000,
        # Paths for saving model and logs
        model_save_path="model/ddqn_1-1_supervised.pth",
        csv_filename="training_results/training_log_ddqn_1-1.csv",
        # Prioritized Replay Buffer parameters
        priority_alpha=0.6,
        priority_beta_start=0.4,
        priority_beta_frames=100000,
        # Parameters for running a trained agent
        runs=50,
        delay=0.05,
        # Parameters for plotting training results
        moving_average_window=300,
        # Training control parameters
        num_episodes=100000,
        save_every=100,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.replay_memory_size = replay_memory_size
        self.target_update_frequency = target_update_frequency

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.model_save_path = model_save_path
        self.csv_filename = csv_filename

        self.priority_alpha = priority_alpha
        self.priority_beta_start = priority_beta_start
        self.priority_beta_frames = priority_beta_frames

        self.runs = runs
        self.delay = delay

        self.moving_average_window = moving_average_window

        self.num_episodes = num_episodes
        self.save_every = save_every
