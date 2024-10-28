import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PGM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PGM, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PGMRLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate, gamma):
        self.model = PGM(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        action = torch.argmax(q_values).item()
        return action

    def update(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_target = q_values.clone()
        q_target[action] = reward + self.gamma * torch.max(next_q_values)
        loss = (q_values - q_target).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Example usage:
if __name__ == "__main__":
    state_dim = 10
    action_dim = 5
    hidden_dim = 20
    learning_rate = 0.001
    gamma = 0.99

    agent = PGMRLAgent(state_dim, action_dim, hidden_dim, learning_rate, gamma)

    state = np.random.rand(state_dim)
    action = agent.select_action(state)
    next_state = np.random.rand(state_dim)
    reward = np.random.rand()

    agent.update(state, action, reward, next_state)
