import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, epsilon=0.1, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.buffer = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > 1000:
            self.buffer.pop(0)

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.target_model(next_state)
        q_target = q_values.clone()
        q_target[action] = reward + self.gamma * torch.max(next_q_values)

        loss = (q_values - q_target).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
