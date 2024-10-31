# moneyverse/self_learning/self_learning_engine.py

import logging
import numpy as np
from moneyverse.algorithms.rl_agent import RLAgent
from moneyverse.algorithms.replay_buffer import ReplayBuffer
from moneyverse.database.db_connection import DatabaseConnection

class SelfLearningEngine:
    """
    Self-learning engine for reinforcement learning-driven strategy improvement.

    Attributes:
    - agent (RLAgent): The reinforcement learning agent responsible for policy optimization.
    - replay_buffer (ReplayBuffer): Stores experiences for training.
    - db (DatabaseConnection): Connection for logging learning progress.
    - logger (Logger): Tracks learning events and performance metrics.
    """

    def __init__(self, agent: RLAgent, db: DatabaseConnection, buffer_size=2000):
        self.agent = agent
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.logger.info("SelfLearningEngine initialized.")

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay buffer.

        Args:
        - state (np.ndarray): Current state.
        - action (int): Action taken.
        - reward (float): Reward received.
        - next_state (np.ndarray): Next state observed.
        - done (bool): Whether the episode ended.
        """
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.store(experience)
        self.logger.debug("Stored experience in replay buffer.")

    def train_from_replay(self, batch_size=32):
        """
        Samples experiences and trains the agent using the replay buffer.

        Args:
        - batch_size (int): Number of experiences to sample for each training step.
        """
        if len(self.replay_buffer) < batch_size:
            self.logger.warning("Not enough samples in replay buffer to train.")
            return

        batch = self.replay_buffer.sample(batch_size)
        loss = self.agent.update_policy(batch)
        self.logger.info(f"Training step completed. Policy loss: {loss}")

    def log_performance(self, performance_metric: float):
        """
        Logs the agent's performance to the database for long-term analysis.

        Args:
        - performance_metric (float): The current performance score of the agent.
        """
        self.db.log_model_performance("RLAgent", performance_metric)
        self.logger.info(f"Logged RL agent performance: {performance_metric}")

    def adapt_strategy(self, market_condition: str):
        """
        Adjusts the agent’s behavior based on the current market condition.

        Args:
        - market_condition (str): The current observed market condition.
        """
        if market_condition == "volatile":
            self.agent.adjust_exploration_rate(0.3)
        elif market_condition == "stable":
            self.agent.adjust_exploration_rate(0.1)
        else:
            self.agent.adjust_exploration_rate(0.2)

        self.logger.info(f"Adapted strategy for market condition: {market_condition}")
