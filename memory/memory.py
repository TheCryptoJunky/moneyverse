import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.buffer = []
        self.seed = np.random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)
