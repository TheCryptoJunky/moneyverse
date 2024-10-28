import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RlAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RlAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the RL agent
agent = RlAgent(state_dim=10, action_dim=3)

# Define the optimizer and loss function
optimizer = optim.Adam(agent.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Autonomous learning capabilities
def learn_from_experience(experiences):
    states, actions, rewards, next_states = experiences
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    # Calculate the target Q-values
    q_values = agent(states)
    next_q_values = agent(next_states)
    target_q_values = rewards + 0.99 * torch.max(next_q_values, dim=1)[0]

    # Calculate the loss
    loss = loss_fn(q_values, target_q_values.unsqueeze(1))

    # Backpropagate the loss and update the agent's parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Example usage:
experiences = np.random.rand(100, 10), np.random.rand(100, 3), np.random.rand(100), np.random.rand(100, 10)
learn_from_experience(experiences)

import numpy as np
import pandas as pd
from typing import Tuple

class QLearningAgent:
    def __init__(self, alpha: float, gamma: float, epsilon: float, num_states: int, num_actions: int):
        """
        Initialize Q-learning agent.

        Args:
        - alpha (float): Learning rate.
        - gamma (float): Discount factor.
        - epsilon (float): Exploration rate.
        - num_states (int): Number of states.
        - num_actions (int): Number of actions.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state: int) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
        - state (int): Current state.

        Returns:
        - action (int): Chosen action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update Q-table using Q-learning update rule.

        Args:
        - state (int): Current state.
        - action (int): Chosen action.
        - reward (float): Received reward.
        - next_state (int): Next state.
        """
        q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, np.argmax(self.q_table[next_state])]
        self.q_table[state, action] = (1 - self.alpha) * q_value + self.alpha * (reward + self.gamma * next_q_value)

    def save_q_table(self, file_path: str) -> None:
        """
        Save Q-table to a file.

        Args:
        - file_path (str): File path to save Q-table.
        """
        np.save(file_path, self.q_table)

    def load_q_table(self, file_path: str) -> None:
        """
        Load Q-table from a file.

        Args:
        - file_path (str): File path to load Q-table.
        """
        self.q_table = np.load(file_path)

def main():
    # Example usage
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, num_states=10, num_actions=3)
    state = 0
    action = agent.choose_action(state)
    reward = 10.0
    next_state = 1
    agent.update_q_table(state, action, reward, next_state)
    agent.save_q_table("q_table.npy")

if __name__ == "__main__":
    main()
