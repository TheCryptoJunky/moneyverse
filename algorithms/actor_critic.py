# /bot/src/rl_agent/actor_critic.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=0)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, epsilon=0.1, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.memory = []

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            action_probs = self.actor(state)
            return torch.argmax(action_probs).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        batch = np.random.choice(self.memory, batch_size, replace=False)
        states = torch.tensor([x[0] for x in batch], dtype=torch.float32)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.int64)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        next_states = torch.tensor([x[3] for x in batch], dtype=torch.float32)
        dones = torch.tensor([x[4] for x in batch], dtype=torch.bool)

        action_probs = self.actor(states)
        action_log_probs = torch.log(action_probs)
        q_values = self.critic(states)
        next_q_values = self.critic(next_states)
        q_targets = q_values.clone()
        q_targets[range(batch_size)] = rewards + 0.99 * next_q_values * (~dones)

        actor_loss = -action_log_probs[range(batch_size), actions] * (q_targets - q_values)
        critic_loss = (q_values - q_targets).pow(2)
        loss = actor_loss.mean() + critic_loss.mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

# Example usage:
# agent = ActorCriticAgent(state_dim=10, action_dim=3)
# state = np.random.rand(10)
# action = agent.act(state)
# agent.remember(state, action, 1.0, np.random.rand(10), False)
# agent.replay(32)
