import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling prioritized experiences.
    
    Attributes:
    - max_size (int): Maximum buffer capacity.
    - buffer (deque): Buffer storing (state, action, reward, next_state, done).
    - priorities (deque): Tracks priority levels for experiences.
    - alpha (float): Degree of prioritization (0 = uniform sampling, 1 = full prioritization).
    """

    def __init__(self, max_size=2000, alpha=0.6):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha

    def store(self, experience, priority=1.0):
        """
        Adds an experience to the buffer with a given priority.

        Args:
        - experience (tuple): (state, action, reward, next_state, done).
        - priority (float): Initial priority for sampling.
        """
        self.buffer.append(experience)
        self.priorities.append(priority ** self.alpha)

    def sample(self, batch_size=32, beta=0.4):
        """
        Samples a batch of experiences using prioritized sampling.

        Args:
        - batch_size (int): Number of experiences to sample.
        - beta (float): Degree of importance-sampling correction.

        Returns:
        - list: Sampled experiences.
        - np.ndarray: Importance-sampling weights.
        """
        scaled_priorities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=scaled_priorities)
        experiences = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * scaled_priorities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        return experiences, weights

    def update_priorities(self, indices, priorities):
        """
        Updates the priorities for experiences.

        Args:
        - indices (list): Indices of experiences to update.
        - priorities (list): New priorities.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha

    def size(self):
        """
        Returns current buffer size.

        Returns:
        - int: Number of experiences.
        """
        return len(self.buffer)
