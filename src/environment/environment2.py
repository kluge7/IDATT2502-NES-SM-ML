import os
import subprocess as sp

import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym.spaces import Box
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class Monitor:
    def __init__(self, width, height, saved_path):
        ffmpeg_path = os.path.join("..", "..", "ffmpeg", "bin", "ffmpeg.exe")
        self.command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}X{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            "60",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "mpeg4",
            saved_path,
        ]
        try:
            self.pipe = sp.Popen(  # noqa: S603
                self.command, stdin=sp.PIPE, stderr=sp.PIPE, executable=ffmpeg_path
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg not found. Ensure ffmpeg is in the specified directory."
            ) from e

    def record(self, image_array):
        if not hasattr(self, "pipe") or self.pipe is None:
            raise RuntimeError(
                "ffmpeg process not initialized. Unable to record video."
            )
        try:
            self.pipe.stdin.write(image_array.tobytes())
        except BrokenPipeError as e:
            print("ffmpeg error:", self.pipe.stderr.read().decode())
            raise RuntimeError("ffmpeg terminated unexpectedly.") from e

    def close(self):
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.wait()


def process_frame(frame):
    if frame is not None:
        # Check if the frame has three channels (e.g., RGB) before converting to grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize and normalize
        frame = cv2.resize(frame, (84, 84)) / 255.0
        return frame.astype(np.float32)
    else:
        # Return a zeroed frame if input is None
        return np.zeros((1, 84, 84), dtype=np.float32)


class CustomReward3(gym.Wrapper):
    def __init__(self, env=None, monitor=None, stuck_threshold=150, stuck_penalty=-10):
        super().__init__(env)
        self.monitor = monitor
        self.stuck_threshold = stuck_threshold
        self.stuck_penalty = stuck_penalty  # Penalty for getting stuck
        self.prev_x_pos = 0
        self.stuck_counter = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # Monitor video recording if enabled
        if self.monitor:
            self.monitor.record(state)

        # Process frame for PPO training
        state = process_frame(state)

        # Reward adjustment for successful or unsuccessful level completion
        if done:
            reward += 10 if info.get("flag_get", False) else -10

        # Penalize idling or getting stuck
        current_x_pos = info.get("x_pos", 0)

        # Increment stuck counter if no progress; reset if moving forward
        if current_x_pos <= self.prev_x_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0  # Reset counter if forward progress is made

        # End episode if stuck for too long
        if self.stuck_counter >= self.stuck_threshold:
            reward += self.stuck_penalty  # Apply penalty for getting stuck
            done = True  # Force end the episode

        # Update previous x position
        self.prev_x_pos = current_x_pos

        return state, reward, done, info

    def reset(self):
        self.prev_x_pos = 0  # Reset x position tracking
        self.stuck_counter = 0  # Reset stuck counter
        return process_frame(self.env.reset())


class CustomSkipFrame2(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=1, shape=(skip, 84, 84), dtype=np.float32
        )
        self.skip = skip
        self.frames = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_frames = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            processed_state = process_frame(state)
            total_reward += reward
            if i >= self.skip // 2:
                last_frames.append(processed_state)
            if done:
                break
        if last_frames:
            max_frame = np.max(np.stack(last_frames), axis=0)
        else:
            max_frame = processed_state
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = max_frame
        return self.frames, total_reward, done, info

    def reset(self):
        initial_frame = process_frame(self.env.reset())
        self.frames = np.stack([initial_frame] * self.skip, axis=0)
        return self.frames


def create_env(map="SuperMarioBros-1-1-v0", output_path=None):
    env = JoypadSpace(gym_super_mario_bros.make(map), SIMPLE_MOVEMENT)
    monitor = (
        Monitor(width=256, height=240, saved_path=output_path) if output_path else None
    )
    env = CustomReward3(env, monitor=monitor)
    env = CustomSkipFrame2(env)
    return env
