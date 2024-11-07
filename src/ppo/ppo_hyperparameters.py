class PPOHyperparameters:
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
        ent_coef=0.01,
        max_grad_norm=0.5,
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
