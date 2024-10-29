# Full file path: moneyverse/rl_agent/marl.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class MARL(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MARL, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MARL_Agent:
    def __init__(self, state_dim, action_dim, replay_memory_size=10000, batch_size=64, lr=0.001, gamma=0.99):
        self.model = MARL(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_rate = 1.0  # Start with full exploration
        self.exploration_decay = 0.995  # Decay factor for exploration rate
        self.min_exploration_rate = 0.01

    def select_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.model.fc3.out_features - 1)
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        action = torch.argmax(q_values).item()
        return action

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience with priority for replay."""
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough experiences to train on

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float32)
            
            q_values = self.model(state)
            next_q_values = self.model(next_state)
            q_value = q_values[action]
            next_q_value = next_q_values.max() if not done else 0  # No future reward if done

            # Bellman Equation Update
            target = reward + self.gamma * next_q_value
            loss = nn.functional.mse_loss(q_value, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Adjust exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
