import collections

import cv2
import gym
import numpy as np


class ActionRepeat(gym.Wrapper):
    """Repeats the action for a specified number of frames (default = 4)."""

    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.buffer = collections.deque(maxlen=2)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        done = False
        for _i in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.buffer.append(observation)
            if done:
                break
        max_frame = np.max(np.stack(self.buffer), axis=0)
        self.frame = max_frame.copy()
        return max_frame, total_reward, done, info


class ResizeAndGrayscale(gym.ObservationWrapper):
    """Resizes and converts the observation to grayscale, outputting 84x84 images."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

        
    def observation(self, observation):
        return ResizeAndGrayscale.convert(observation)


    @staticmethod
    def convert(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            raise ValueError("Unknown resolution.")
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        cropped_img = resized_img[18:102, :]
        processed_frame = np.reshape(cropped_img, [84, 84, 1])
        return processed_frame.astype(np.uint8)


class ConvertToTensor(gym.ObservationWrapper):
    """Converts observations to tensor format suitable for PyTorch."""

    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ObservationBuffer(gym.ObservationWrapper):
    """Maintains a sliding window of the last n observations."""

    def __init__(self, env, n_steps, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype,
        )

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class NormalizePixels(gym.ObservationWrapper):
    """Normalizes pixel values in the observation to a range of [0, 1]."""

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
