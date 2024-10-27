import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mev_strategy import MEVStrategy

class RLAgent:
    def __init__(self, strategy, device):
        self.strategy = strategy
        self.device = device
        self.policy_network = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        ).to(device)
        self.value_network = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        q_values = self.policy_network(state)
        action = torch.argmax(q_values)
        return action.item()

    def update_policy(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        q_values = self.policy_network(state)
        next_q_values = self.policy_network(next_state)
        q_target = q_values.clone()
        q_target[action] = reward + 0.99 * next_q_values.max()
        loss = (q_values - q_target).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_value(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        value = self.value_network(state)
        return value.item()

    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = self._build_q_network()
        self.target_q_network = self._build_q_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.writer = SummaryWriter()

    def _build_q_network(self):
        q_network = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        return q_network

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.q_network(state)
        next_q_values = self.target_q_network(next_state)
        q_value = q_values[action]
        next_q_value = next_q_values.max()

        loss = (q_value - (reward + self.gamma * next_q_value)) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('Loss', loss.item())

    def sync_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
