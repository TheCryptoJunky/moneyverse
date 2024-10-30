import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling experiences with prioritization.

    Attributes:
    - max_size (int): Maximum buffer size.
    - buffer (deque): Stores experiences as (state, action, reward, next_state, done).
    - priorities (list): Tracks priorities of experiences for prioritized sampling.
    """

    def __init__(self, max_size=2000, alpha=0.6):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha  # Controls degree of prioritization (0 = uniform sampling)

    def store(self, experience, priority=1.0):
        """
        Stores an experience with an initial priority.

        Args:
        - experience (tuple): The experience tuple (state, action, reward, next_state, done).
        - priority (float): Initial priority of the experience.
        """
        self.buffer.append(experience)
        self.priorities.append(priority ** self.alpha)

    def sample(self, batch_size=32, beta=0.4):
        """
        Samples a batch of experiences using prioritized sampling.

        Args:
        - batch_size (int): Number of experiences to sample.
        - beta (float): Controls the degree of importance-sampling correction.

        Returns:
        - list: Batch of experiences.
        - np.ndarray: Importance-sampling weights for each experience in the batch.
        """
        scaled_priorities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=scaled_priorities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance-sampling weights to correct for bias in prioritized replay
        weights = (len(self.buffer) * scaled_priorities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights to keep them within a manageable range

        return experiences, weights

    def update_priorities(self, indices, priorities):
        """
        Updates priorities for sampled experiences after learning.

        Args:
        - indices (list): Indices of the sampled experiences.
        - priorities (list): New priorities for the experiences.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha

    def size(self):
        """
        Returns the current size of the buffer.

        Returns:
        - int: Number of stored experiences.
        """
        return len(self.buffer)
