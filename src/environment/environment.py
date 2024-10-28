import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.environment.wrappers import (
    ActionRepeat,
    ConvertToTensor,
    NormalizePixels,
    ObservationBuffer,
    ResizeAndGrayscale,
)


def create_env(map="SuperMarioBros-v0", action_repeat=4):
    """Sets up the Super Mario Bros environment with customized wrappers."""
    env = ActionRepeat(gym_super_mario_bros.make(map), action_repeat)
    env = ResizeAndGrayscale(env)
    env = ConvertToTensor(env)
    env = ObservationBuffer(env, 4)
    env = NormalizePixels(env)
    return JoypadSpace(env, SIMPLE_MOVEMENT)
