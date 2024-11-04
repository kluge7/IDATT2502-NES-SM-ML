import numpy as np
import gym

class CustomReward(gym.RewardWrapper):
    """A custom reward wrapper that modifies and clips rewards based on game score and level completion."""

    def __init__(self, env=None, monitor=None):
        super().__init__(env)
        self.monitor = monitor
        self.curr_score = 0

    def step(self, action):
        # Perform the action in the environment
        state, reward, done, info = self.env.step(action)

        # Optionally record the state if a monitor is provided
        if self.monitor:
            self.monitor.record(state)

        # Update the reward based on score differences
        score_diff = info.get("score", 0) - self.curr_score
        reward += score_diff / 40.0
        self.curr_score = info.get("score", 0)

        # Additional rewards/penalties for completion or failure
        if done:
            if info.get("flag_get", False):
                reward += 50  # Reward for completing the level
            else:
                reward -= 50  # Penalty for failing the level

        # Clip the reward to the range [-1, 1]
        reward = np.clip(reward, -1, 1)

        return state, reward, done, info

    def reset(self, **kwargs):
        # Reset the score tracker
        self.curr_score = 0
        return self.env.reset(**kwargs)
