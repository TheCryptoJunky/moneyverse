# Deep Q-Network (DQN) Agent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = Memory()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        return model

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        q_values = self.model(state)
        next_q_values = self.target_model(next_state)
        q_target = q_values.clone()
        q_target[action] = reward + self.gamma * torch.max(next_q_values)
        loss = (q_values - q_target) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Changes:
# 1. Added `epsilon` parameter to control exploration-exploitation trade-off.
# 2. Implemented `select_action` method to select actions based on epsilon-greedy policy.
# 3. Updated `update` method to use target model for calculating next Q-values.
