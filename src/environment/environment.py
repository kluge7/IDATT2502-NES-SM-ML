import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.environment.wrappers import (
    ConvertToTensor,
    CustomReward,
    FrameSkipper,
    Monitor,
    NormalizePixels,
    ObservationBuffer,
    ResizeAndGrayscale,
)


def create_env(
    map="SuperMarioBros-v0",
    skip=4,
    output_path=None,
    actions=SIMPLE_MOVEMENT,
    stages=None,
):
    """Sets up the Super Mario Bros environment with customized wrappers."""
    # env = JoypadSpace(gym_super_mario_bros.make(map, stages), actions)

    env = gym.make("SuperMarioBrosRandomStages-v0", stages=["1-1", "4-1"])
    env = JoypadSpace(env, actions)

    if output_path is not None:
        monitor = Monitor(width=256, height=240, saved_path=output_path)
    else:
        monitor = None
    env = CustomReward(env, monitor=monitor)
    env = FrameSkipper(env, skip=skip)
    env = ResizeAndGrayscale(env)
    env = ConvertToTensor(env)
    env = ObservationBuffer(env, 4)
    env = NormalizePixels(env)
    return env
