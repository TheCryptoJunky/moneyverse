import numpy as np
import logging
from .dqn import DQNAgent
from .replay_buffer import ReplayBuffer
from ..database.db_connection import DatabaseConnection

class MARLAgent:
    """
    Multi-Agent Reinforcement Learning (MARL) agent coordinating actions across multiple DQN agents.
    
    Attributes:
    - db (DatabaseConnection): Database connection for logging performance.
    - agents (list): List of individual DQN agents.
    - replay_buffer (ReplayBuffer): Shared experience buffer for all agents.
    - epsilon (float): Exploration rate for coordinated epsilon-greedy strategy.
    """

    def __init__(self, num_agents: int, state_size: int, action_size: int, db: DatabaseConnection):
        self.db = db
        self.replay_buffer = ReplayBuffer(max_size=5000)
        self.agents = [DQNAgent(state_size, action_size, db) for _ in range(num_agents)]
        self.epsilon = 1.0
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized MARL with {num_agents} agents.")

    def select_joint_action(self, states: list) -> list:
        """
        Selects actions for each agent in a coordinated manner, factoring in exploration.

        Args:
        - states (list): List of states, one for each agent.
        
        Returns:
        - list: Actions selected by each agent.
        """
        joint_action = []
        for agent, state in zip(self.agents, states):
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(agent.action_size)  # Exploration
                self.logger.debug(f"Agent exploring with action {action}.")
            else:
                action = agent.act(state)  # Exploitation
                self.logger.debug(f"Agent exploiting with action {action}.")
            joint_action.append(action)
        
        self.logger.info(f"Joint action selected: {joint_action}")
        return joint_action

    def store_experience(self, states, actions, rewards, next_states, dones):
        """
        Stores experiences for all agents in the shared replay buffer.

        Args:
        - states (list): Current states for each agent.
        - actions (list): Actions taken by each agent.
        - rewards (list): Rewards received by each agent.
        - next_states (list): Next states for each agent.
        - dones (list): Whether each agent’s episode has ended.
        """
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.replay_buffer.store((state, action, reward, next_state, done))
        self.logger.info("Stored joint experience in replay buffer.")

    def train_agents(self, batch_size=32):
        """
        Trains each agent with a shared experience replay buffer.

        Args:
        - batch_size (int): Number of experiences to sample for training.
        """
        if len(self.replay_buffer) < batch_size:
            self.logger.warning("Insufficient experiences for MARL training.")
            return

        experiences, weights = self.replay_buffer.sample(batch_size)
        for agent in self.agents:
            for state, action, reward, next_state, done in experiences:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()

        # Decay epsilon to balance exploration with exploitation over time
        self.epsilon = max(0.01, self.epsilon * 0.995)
        self.logger.info(f"Trained agents; epsilon decayed to {self.epsilon}.")

    def log_performance(self, episode: int, rewards: list):
        """
        Logs the performance of each agent for tracking and evaluation.
        
        Args:
        - episode (int): Current episode number.
        - rewards (list): Rewards collected by each agent.
        """
        for i, reward in enumerate(rewards):
            self.db.log_performance(agent_id=i, episode=episode, reward=reward)
            self.logger.info(f"Agent {i} performance logged for episode {episode} with reward {reward}.")
