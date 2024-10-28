# moneyverse/rl_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RLAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_network(state)
        action = torch.argmax(action_probs)
        return action.item()

    def update_policy(self, state, action, reward):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)

        self.optimizer.zero_grad()
        loss = -reward * self.policy_network(state)[action]
        loss.backward()
        self.optimizer.step()
