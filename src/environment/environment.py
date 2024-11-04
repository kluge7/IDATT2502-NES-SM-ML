import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.environment.wrappers import (
    ActionRepeat,
    ConvertToTensor,
    CustomReward,
    Monitor,
    NormalizePixels,
    ObservationBuffer,
    ResizeAndGrayscale,
)


def create_env(map="SuperMarioBros-1-1-v0", action_repeat=4, output_path=None):
    """Sets up the Super Mario Bros environment with customized wrappers."""
    env = JoypadSpace(gym_super_mario_bros.make(map), SIMPLE_MOVEMENT)
    if output_path is not None:
        monitor = Monitor(width=256, height=240, saved_path=output_path)
    else:
        monitor = None
    env = CustomReward(env, monitor=monitor)
    env = ActionRepeat(env, action_repeat)
    env = ResizeAndGrayscale(env)
    env = ConvertToTensor(env)
    env = ObservationBuffer(env, 4)
    env = NormalizePixels(env)
    return env
