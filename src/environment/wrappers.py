from collections import deque

import cv2
import gym
import numpy as np
from gym import ObservationWrapper


# TODO: Adapt if needed for RL
class SkipFrame(gym.Wrapper):
    """Wrapper to skip a specified number of frames."""

    def __init__(self, env, skip) -> None:
        """Initialize the SkipFrame wrapper.

        Args:
        ----
            env (gym.Env): The environment to wrap.
            skip (int): The number of frames to skip.

        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action for a given number of frames and accumulate rewards.

        Args:
        ----
            action: The action to be repeated over the skipped frames.

        Returns:
        -------
            state: The observation after the skipped frames.
            total_reward: The total accumulated reward during the skipped frames.
            done: Whether the episode has ended.
            info: Additional information from the environment.

        """
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, info


class GrayScaleObservation(ObservationWrapper):
    """Convert observations from RGB to grayscale."""

    def __init__(self, env) -> None:
        """Initialize the GrayScaleObservation wrapper.

        Args:
        ----
            env (gym.Env): The environment to wrap.

        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_space.shape[0], self.observation_space.shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        """Convert an RGB observation to grayscale.

        Args:
        ----
            observation: The RGB observation from the environment.

        Returns:
        -------
            The grayscale version of the observation.

        """
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)


class ResizeObservation(ObservationWrapper):
    """Resize observations to a given shape."""

    def __init__(self, env, shape) -> None:
        """Initialize the ResizeObservation wrapper.

        Args:
        ----
            env (gym.Env): The environment to wrap.
            shape (tuple): The desired shape for resizing the observation (height, width).

        """
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[0], shape[1], 1), dtype=np.uint8
        )

    def observation(self, observation):
        """Resize the observation to the specified shape.

        Args:
        ----
            observation: The observation from the environment.

        Returns:
        -------
            The resized observation.

        """
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation


class PixelNormalize(ObservationWrapper):
    """Normalize pixel values from 0-255 to 0-1."""

    def __init__(self, env):
        """Initialize the PixelNormalize wrapper.

        Args:
        ----
            env (gym.Env): The environment to wrap.

        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        """Normalize the pixel values of the observation to the range [0, 1].

        Args:
        ----
            obs: The observation from the environment.

        Returns:
        -------
            The normalized observation.

        """
        return obs.astype(np.float32) / 255.0


class FrameStack(gym.Wrapper):
    """Stack the last `k` frames to give the agent a sense of motion."""

    def __init__(self, env, k):
        """Initialize the FrameStack wrapper.

        Args:
        ----
            env (gym.Env): The environment to wrap.
            k (int): The number of frames to stack.

        """
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        """Reset the environment and stack `k` initial frames.

        Returns:
            The stacked frames as the initial observation.

        """
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_observation()

    def step(self, action):
        """Take a step in the environment and stack the frames.

        Args:
            action: The action to be taken in the environment.

        Returns:
            obs: The stacked frames after the action.
            reward: The reward from the action.
            done: Whether the episode has ended.
            info: Additional information from the environment.

        """
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """Get the stacked observation.

        Returns:
        -------
            The concatenated frames.

        """
        return np.concatenate(list(self.frames), axis=-1)


class FrameToTensor(ObservationWrapper):
    """Convert frames to tensors and move the channel axis to the front."""

    def __init__(self, env):
        """Initialize the FrameToTensor wrapper.

        Args:
        ----
            env (gym.Env): The environment to wrap.

        """
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32,
        )

    def observation(self, observation):
        """Move the channel axis to the front for compatibility with PyTorch.

        Args:
        ----
            observation: The observation from the environment.

        Returns:
        -------
            The observation with the channel axis moved to the front.

        """
        return np.moveaxis(observation, -1, 0)  # Move channel axis


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to the range [-1, 1]."""

    def reward(self, reward):
        """Clip the reward to the range [-1, 1].

        Args:
        ----
            reward: The raw reward from the environment.

        Returns:
        -------
            The clipped reward.

        """
        return np.sign(reward)
