import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
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


def create_env(map="SuperMarioBros-1-1-v0", skip=4, output_path=None):
    """Sets up the Super Mario Bros environment with customized wrappers."""
    env = JoypadSpace(gym_super_mario_bros.make(map), COMPLEX_MOVEMENT)
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
