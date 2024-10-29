# Full file path: moneyverse/rl_agent/rl_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.reward_calculator import calculate_reward  # Integrate reward calculation
from database.async_db_handler import AsyncDBHandler
from cryptography.fernet import Fernet

# Initialize encryption key for secure model checkpoints if necessary
cipher = Fernet(Fernet.generate_key())
db_handler = AsyncDBHandler()

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

# Neural network-based RL agent initialization
agent = RlAgent(state_dim=10, action_dim=3)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Replay buffer for experience replay
replay_buffer = []

# Enhanced learning function with experience replay
def learn_from_experience():
    if len(replay_buffer) < 64:
        return

    batch = np.random.choice(replay_buffer, 64, replace=False)
    states, actions, rewards, next_states = zip(*batch)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    # Q-value calculations
    q_values = agent(states)
    next_q_values = agent(next_states)
    target_q_values = rewards + 0.99 * torch.max(next_q_values, dim=1)[0]
    loss = loss_fn(q_values, target_q_values.unsqueeze(1))

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Q-Learning agent with adjustable epsilon decay
class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, num_states, num_actions):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        # Update Q-table
        q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, np.argmax(self.q_table[next_state])]
        self.q_table[state, action] = (1 - self.alpha) * q_value + self.alpha * (reward + self.gamma * next_q_value)

    def update_epsilon(self, min_epsilon=0.01, decay_rate=0.995):
        # Epsilon decay
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    async def save_q_table(self, table_name="q_table"):
        encrypted_data = cipher.encrypt(self.q_table.dumps().encode())
        await db_handler.execute("INSERT INTO model_checkpoints (table_name, data) VALUES ($1, $2)", table_name, encrypted_data)

    async def load_q_table(self, table_name="q_table"):
        data = await db_handler.fetch("SELECT data FROM model_checkpoints WHERE table_name = $1", table_name)
        self.q_table = np.loads(cipher.decrypt(data[0]["data"]).decode())

# Example usage
if __name__ == "__main__":
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, num_states=10, num_actions=3)
    state, action, reward, next_state = 0, agent.choose_action(0), 10, 1
    agent.update_q_table(state, action, reward, next_state)
