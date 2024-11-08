import random
from collections import deque, namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition.

        Args:
            *args: A single transition consisting of state, action, reward, next_state, and done flag.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        """Randomly samples a batch of transitions from the replay memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            list[Transition]: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
