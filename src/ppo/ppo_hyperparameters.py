class PPOHyperparameters:
    def __init__(
        self,
        timesteps_per_batch=4000,
        max_timesteps_per_episode=600,
        gamma=0.99,
        n_updates_per_iteration=5,
        clip=0.2,
        lr=1e-4,
        dynamic_lr=False,
        min_lr_limit=0.000001,
        num_minibatches=5,
        lam=0.98,
        ent_coef=0.1,
        max_grad_norm=0.5,
        target_kl=0.02,
        save_freq=1,
        render=False,
        specification="SuperMarioBros-1-1-v0",
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
        self.save_freq = save_freq
        self.render = render
        self.specification = specification
        self.model_path = model_path
        self.training_result_path = training_result_path
        self.episode_result_path = episode_result_path
