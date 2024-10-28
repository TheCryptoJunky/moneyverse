# Policy Gradient Method (PGM) Agent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PGMAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = Memory()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=1)
        )
        return model

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probabilities = self.model(state)
        return torch.multinomial(probabilities, 1).item()

    def update(self, state, action, reward):
        state = torch.tensor(state, dtype=torch.float32)
        probabilities = self.model(state)
        log_prob = torch.log(probabilities[action])
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Changes:
# 1. Implemented `select_action` method to select actions based on policy.
# 2. Updated `update` method to use policy gradient update rule.
