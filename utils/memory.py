# Memory

import numpy as np

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

# Changes:
# 1. Implemented `Memory` class to store experiences.
