# moneyverse/algorithms/replay_buffer.py

import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Memory-efficient replay buffer for experience sampling in reinforcement learning.

    Attributes:
    - buffer (deque): Fixed-size buffer storing experiences in FIFO order.
    - max_size (int): Maximum size of the buffer.
    """

    def __init__(self, max_size=2000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def store(self, experience):
        """
        Stores an experience in the buffer.

        Args:
        - experience (tuple): (state, action, reward, next_state, done).
        """
        self.buffer.append(experience)

    def sample(self, batch_size=32):
        """
        Samples a random batch of experiences from the buffer.

        Args:
        - batch_size (int): Number of experiences to sample.

        Returns:
        - list: A list of randomly sampled experiences.
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        return batch

    def __len__(self):
        """
        Returns the current size of the buffer.
        
        Returns:
        - int: Number of experiences in the buffer.
        """
        return len(self.buffer)
