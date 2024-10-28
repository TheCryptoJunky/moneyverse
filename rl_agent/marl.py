import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, state_dim, action_dim):
        self.model = MARL(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        action = torch.argmax(q_values).item()
        return action

    def update(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_value = q_values[action]
        next_q_value = next_q_values.max()

        loss = (q_value - (reward + 0.99 * next_q_value)) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
