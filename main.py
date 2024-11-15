import argparse
import os

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from src.environment.environment import create_env
from src.ppo.ppo_agent import PPOAgent
from src.ppo.ppo_hyperparameters import PPOHyperparameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO trainer")

    # General options for main training/testing
    parser.add_argument(
        "option",
        nargs="?",
        default="train",
        type=str,
        help="Select an option: 'train' for PPO training, 'test' for PPO testing on a model",
    )

    # General options for main method (DDQN/PPO/Supervised)
    parser.add_argument(
        "--method",
        nargs="?",
        default="PPO",
        type=str,
        help="Select an method: 'PPO' or 'DDQN'",
    )

    parser.add_argument(
        "--world", type=int, default=1, help="World number for the environment"
    )
    parser.add_argument(
        "--stage", type=int, default=1, help="Stage number within the world"
    )
    parser.add_argument(
        "--env_version", type=str, default="v0", help="Environment version"
    )
    parser.add_argument(
        "--render",
        type=bool,
        default=True,
        help="Render the environment during training",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=3_000_000,
        help="Total timesteps for training",
    )

    parser.add_argument(
        "--actor_load_path",
        type=str,
        default="",
        help="The path of the actor model to load",
    )
    parser.add_argument(
        "--critic_load_path",
        type=str,
        default="",
        help="The path of the critic model to load",
    )

    parser.add_argument(
        "--record", type=bool, default=False, help="Whether to record test gameplay"
    )
    # parser.add_argument("--output_file_name", type=str, default=None, help="Name of mp4 video")
    parser.add_argument(
        "--test_episode_number", type=int, default=1, help="Number of episodes to test"
    )

    # Hyperparameters for PPO
    parser.add_argument(
        "--timesteps_per_batch",
        type=int,
        default=2000,
        help="Timesteps per training batch",
    )
    parser.add_argument(
        "--max_timesteps_per_episode",
        type=int,
        default=400,
        help="Max timesteps per episode",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.90, help="Discount factor for future rewards"
    )
    parser.add_argument(
        "--n_updates_per_iteration",
        type=int,
        default=5,
        help="Updates per PPO iteration",
    )
    parser.add_argument(
        "--clip", type=float, default=0.2, help="Clipping parameter for policy updates"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--dynamic_lr",
        type=bool,
        default=False,
        help="Enable dynamic learning rate adjustment",
    )
    parser.add_argument(
        "--min_lr_limit",
        type=float,
        default=0.000001,
        help="Minimum limit for learning rate",
    )
    parser.add_argument(
        "--num_minibatches",
        type=int,
        default=5,
        help="Number of mini-batches for PPO updates",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.98,
        help="Lambda for Generalized Advantage Estimation (GAE)",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for exploration",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--kl_divergence",
        type=bool,
        default=True,
        help="Enable KL divergence check for stopping",
    )
    parser.add_argument(
        "--target_kl", type=float, default=0.02, help="Target KL divergence threshold"
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=1,
        help="Model save frequency in iterations",
    )

    # Paths and files
    parser.add_argument(
        "--specification",
        type=str,
        default="SuperMarioBros-1-1-v0",
        help="Environment specification",
    )
    parser.add_argument(
        "--model_actor",
        type=str,
        default="ppo_actor.pth",
        help="Actor model checkpoint file",
    )
    parser.add_argument(
        "--model_critic",
        type=str,
        default="ppo_critic.pth",
        help="Critic model checkpoint file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/ppo/model",
        help="Directory for saving model checkpoints",
    )
    parser.add_argument(
        "--training_result_path",
        type=str,
        default="src/ppo/training_result",
        help="Training results directory",
    )
    parser.add_argument(
        "--episode_result_path",
        type=str,
        default="src/ppo/episode_result",
        help="Episode results directory",
    )

    args = parser.parse_args()

    hyperparameters = PPOHyperparameters(
        timesteps_per_batch=args.timesteps_per_batch,
        max_timesteps_per_episode=args.max_timesteps_per_episode,
        gamma=args.gamma,
        n_updates_per_iteration=args.n_updates_per_iteration,
        clip=args.clip,
        lr=args.lr,
        dynamic_lr=args.dynamic_lr,
        min_lr_limit=args.min_lr_limit,
        num_minibatches=args.num_minibatches,
        lam=args.lam,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        kl_divergence=args.kl_divergence,
        target_kl=args.target_kl,
        save_frequency=args.save_frequency,
        render=args.render,
        specification=args.specification,
        model_actor=args.model_actor,
        model_critic=args.model_critic,
        model_path=args.model_path,
        training_result_path=args.training_result_path,
        episode_result_path=args.episode_result_path,
    )

    world, stage, env_version = args.world, args.stage, args.env_version
    specification = f"SuperMarioBros-{world}-{stage}-{env_version}"

    if args.option == "train":
        if args.method == "PPO":
            print("Training PPO model...")
            env = create_env(map=specification, skip=4, actions=COMPLEX_MOVEMENT)
            agent = PPOAgent(env, hyperparameters)
            if args.actor_load_path and args.critic_load_path:
                agent.load_networks(
                    actor_path=args.actor_load_path, critic_path=args.critic_load_path
                )
            agent.train(max_timesteps=args.total_timesteps)

        if args.method == "DDQN":
            print("Training DDQN model...")
            # ADD ddqn training method

    if args.option == "test":
        if args.method == "PPO":
            print("Testing PPO model...")
            if args.record:
                os.makedirs("src/ppo/output", exist_ok=True)
                output_path = f"src/ppo/output/{specification}.mp4"
            else:
                output_path = None
            env = create_env(
                map=specification,
                skip=4,
                actions=COMPLEX_MOVEMENT,
                output_path=output_path,
            )
            agent = PPOAgent(env, hyperparameters)
            if args.actor_load_path and args.critic_load_path:
                agent.load_networks(
                    actor_path=args.actor_load_path, critic_path=args.critic_load_path
                )
            agent.test(episodes=args.test_episode_number)

        if args.method == "DDQN":
            print("Testing DDQN model...")
            # Add ddqn testing method
