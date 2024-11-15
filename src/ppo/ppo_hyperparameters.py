class PPOHyperparameters:
    """Defines hyperparameters for training a Proximal Policy Optimization (PPO) agent.

    Attributes:
        timesteps_per_batch (int): Number of timesteps collected in each batch for training.
        max_timesteps_per_episode (int): Maximum timesteps allowed per episode.

        gamma (float): Discount factor for future rewards, controlling the importance of long-term vs. immediate rewards.
        n_updates_per_iteration (int): Number of PPO update iterations per batch of collected data.
        clip (float): Clipping parameter to limit policy updates, stabilizing training by preventing large policy changes.
        lr (float): Learning rate for optimizers (actor and critic).
        dynamic_lr (bool): Flag to enable dynamic learning rate adjustment during training.
        min_lr_limit (float): Minimum limit for learning rate when `dynamic_lr` is enabled.
        num_minibatches (int): Number of mini-batches used to split each batch of data for PPO updates.
        lam (float): Lambda parameter for Generalized Advantage Estimation (GAE), controlling bias-variance tradeoff.
        ent_coef (float): Entropy coefficient to encourage exploration by adding entropy to the policy loss.
        max_grad_norm (float): Maximum value for gradient clipping to stabilize training by limiting gradient magnitude.
        kl_divergence (bool): Flag to enable Kullback-Leibler (KL) divergence checking for early stopping.
        target_kl (float): Target KL divergence to stop updates early if policy changes too much.

        save_freq (int): Frequency (in iterations) for saving model checkpoints.
        render (bool): Flag to render the environment during training for visualization.

        tensorboard_log_dir (str): Directory path to save TensorBoard logs for training visualization.
        specification (str): Environment specification (default set for Super Mario Bros environment).
        model_actor (str): Filename for saving the actor model checkpoint.
        model_critic (str): Filename for saving the critic model checkpoint.
        model_path (str): Directory path to save model checkpoints.
        training_result_path (str): Directory path to save training results (e.g., CSV files).
        episode_result_path (str): Directory path to save episode-level results (e.g., CSV files).

    """

    def __init__(
        self,
        timesteps_per_batch=2000,
        max_timesteps_per_episode=400,
        # Algorithm
        gamma=0.90,
        n_updates_per_iteration=5,
        clip=0.2,
        lr=1e-4,
        dynamic_lr=False,
        min_lr_limit=0.000001,
        num_minibatches=5,
        lam=0.98,
        ent_coef=0.1,
        max_grad_norm=0.5,
        kl_divergence=True,
        target_kl=0.02,
        save_frequency=1,
        render=False,
        # Paths and default file names
        tensorboard_log_dir="runs/ppo_training",
        specification="SuperMarioBros-1-1-v0",
        model_actor="ppo_actor.pth",
        model_critic="ppo_critic.pth",
        model_path="model",
        training_result_path="training_result",
        episode_result_path="episode_result",
    ):
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode

        self.gamma = gamma
        self.n_updates_per_iteration = n_updates_per_iteration
        self.clip = clip
        self.lr = lr
        self.dynamic_lr = dynamic_lr
        self.min_lr_limit = min_lr_limit
        self.num_minibatches = num_minibatches
        self.lam = lam
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.kl_divergence = kl_divergence
        self.target_kl = target_kl

        self.save_freq = save_frequency
        self.render = render

        self.tensorboard_log_dir = tensorboard_log_dir
        self.specification = specification
        self.model_actor = model_actor
        self.model_critic = model_critic
        self.model_path = model_path
        self.training_result_path = training_result_path
        self.episode_result_path = episode_result_path
