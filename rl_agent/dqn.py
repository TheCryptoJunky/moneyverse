# Full file path: moneyverse/rl_agent/dqn.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Experience replay buffer
        self.memory = deque(maxlen=2000)

        # Models and optimizer
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in the replay buffer with all details."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def experience_replay(self):
        """Train on a batch of experiences sampled from the memory buffer."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            q_values = self.model(state)
            next_q_values = self.target_model(next_state)
            
            target = q_values.clone()
            target[action] = reward + (self.gamma * torch.max(next_q_values) * (1 - int(done)))
            
            loss = nn.functional.mse_loss(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping for stability
            self.optimizer.step()

        # Adjust epsilon for exploration-exploitation trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sync_target_model(self):
        """Synchronize weights from the main model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path="dqn_model.pth"):
        """Saves the model to the specified path."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="dqn_model.pth"):
        """Loads the model from the specified path."""
        self.model.load_state_dict(torch.load(path))
        self.sync_target_model()
