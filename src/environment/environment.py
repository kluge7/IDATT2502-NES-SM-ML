import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import (
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    PixelNormalize,
    FrameStack,
    FrameToTensor,
    ClipRewardEnv,
)


def create_env(map="SuperMarioBros-v0", skip=4):
    """Create a Super Mario environment with appropriate wrappers for RL."""
    env = gym.make(map)
    env = SkipFrame(env, skip=skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = PixelNormalize(env)
    env = FrameStack(env, 4)
    env = FrameToTensor(env)
    env = ClipRewardEnv(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env


env = create_env()

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
