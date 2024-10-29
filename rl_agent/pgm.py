# Full file path: moneyverse/rl_agent/pgm.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class PGMAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, entropy_coef=0.01):
        """
        Policy Gradient Agent with enhancements for adaptive exploration and entropy regularization.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for rewards.
            entropy_coef (float): Coefficient for entropy regularization to encourage exploration.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = []  # For experience replay

    def _build_model(self):
        """Build the policy model."""
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
        """Selects an action using the policy's probability distribution."""
        state = torch.tensor(state, dtype=torch.float32)
        probabilities = self.model(state)
        action = torch.multinomial(probabilities, 1).item()
        return action, probabilities

    def store_experience(self, experience):
        """Stores experience for replay-based learning."""
        self.memory.append(experience)
        if len(self.memory) > 1000:  # Limit memory size
            self.memory.pop(0)

    def update_policy(self, experiences):
        """
        Update policy with experience samples using policy gradient.
        
        Args:
            experiences (list): List of experiences for batch training.
        """
        losses = []
        for state, action, reward in experiences:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probabilities = self.model(state_tensor)
            log_prob = torch.log(probabilities[action])
            entropy = -torch.sum(probabilities * torch.log(probabilities))  # Entropy for exploration
            loss = -log_prob * reward - self.entropy_coef * entropy  # Policy gradient with entropy
            losses.append(loss)

        # Backpropagate loss
        total_loss = torch.stack(losses).sum()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def train_from_memory(self):
        """Trains from stored experiences (replay buffer) periodically."""
        if len(self.memory) > 32:
            sample = random.sample(self.memory, 32)
            self.update_policy(sample)

# Enhancements:
# 1. Entropy regularization added to encourage exploration.
# 2. Experience replay mechanism for stability.
# 3. Adaptive exploration through entropy tuning.
