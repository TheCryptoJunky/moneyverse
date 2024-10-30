import numpy as np
import logging
from .replay_buffer import ReplayBuffer

class ReinforcementAgent:
    """
    Reinforcement Learning Agent for optimizing trading strategy prioritization and decision-making.
    
    Attributes:
    - replay_buffer (ReplayBuffer): Experience replay buffer for training.
    - action_values (dict): Tracks performance metrics for each strategy.
    - epsilon (float): Exploration-exploitation trade-off parameter.
    """

    def __init__(self, epsilon=0.1):
        self.replay_buffer = ReplayBuffer()
        self.action_values = {}
        self.epsilon = epsilon
        self.logger = logging.getLogger(__name__)

    def prioritize_strategies(self, strategies: list) -> list:
        """
        Prioritize strategies using epsilon-greedy selection to balance exploration and exploitation.
        
        Args:
        - strategies (list): List of strategy names.
        
        Returns:
        - list: Ordered list of strategy names based on prioritization.
        """
        if np.random.rand() < self.epsilon:
            prioritized = np.random.permutation(strategies).tolist()
            self.logger.info("Exploration step: Random strategy prioritization.")
        else:
            prioritized = sorted(strategies, key=lambda s: self.action_values.get(s, 0), reverse=True)
            self.logger.info("Exploitation step: Prioritizing strategies by past performance.")
        return prioritized

    def update_strategy_performance(self, strategy_name: str, reward: float):
        """
        Updates the Q-value for a strategy based on the received reward.
        
        Args:
        - strategy_name (str): Name of the strategy.
        - reward (float): Reward received for the strategy's performance.
        """
        current_value = self.action_values.get(strategy_name, 0)
        updated_value = current_value + 0.1 * (reward - current_value)
        self.action_values[strategy_name] = updated_value
        self.logger.info(f"Updated Q-value for {strategy_name}: {updated_value}")

    def add_experience(self, experience):
        """
        Adds an experience to the replay buffer for training.
        
        Args:
        - experience: Tuple (state, action, reward, next_state).
        """
        self.replay_buffer.store(experience)
        self.logger.info("Added new experience to replay buffer.")

    def train(self, batch_size=32):
        """
        Trains the agent using a sample from the replay buffer.
        
        Args:
        - batch_size (int): Number of experiences to sample for training.
        """
        if self.replay_buffer.size() < batch_size:
            self.logger.warning("Not enough experiences in buffer to train.")
            return

        batch = self.replay_buffer.sample(batch_size)
        for state, action, reward, next_state in batch:
            current_value = self.action_values.get(action, 0)
            updated_value = current_value + 0.1 * (reward + np.max([self.action_values.get(a, 0) for a in next_state]) - current_value)
            self.action_values[action] = updated_value
            self.logger.info(f"Trained on action {action}, updated Q-value: {updated_value}")
