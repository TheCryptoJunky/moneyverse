# moneyverse/self_learning/self_learning_engine.py

import logging
from typing import Dict, Callable
import numpy as np

class SelfLearningEngine:
    """
    Facilitates self-learning for optimization of strategies through reinforcement learning (RL).

    Attributes:
    - strategies (dict): Registered strategies with their associated reward functions.
    - learning_rate (float): Controls the weight of updates in the Q-learning algorithm.
    - discount_factor (float): Discount factor for future rewards in RL.
    - q_table (dict): Q-learning table storing values for state-action pairs.
    - logger (Logger): Logs learning process and adjustments.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # {(state, action): q_value}
        self.strategies = {}  # {strategy_name: reward_function}
        self.logger = logging.getLogger(__name__)
        self.logger.info("SelfLearningEngine initialized with Q-learning parameters.")

    def register_strategy(self, strategy_name: str, reward_function: Callable):
        """
        Registers a strategy with its reward function to the self-learning engine.

        Args:
        - strategy_name (str): Name of the strategy.
        - reward_function (callable): Function that calculates reward based on performance metrics.
        """
        self.strategies[strategy_name] = reward_function
        self.logger.info(f"Registered strategy {strategy_name} with self-learning engine.")

    def calculate_reward(self, strategy_name: str, state: str) -> float:
        """
        Calculates the reward for a strategy based on the provided reward function.

        Args:
        - strategy_name (str): Name of the strategy.
        - state (str): Current state of the strategy.

        Returns:
        - float: Calculated reward based on the strategy's performance.
        """
        reward_func = self.strategies.get(strategy_name)
        if not reward_func:
            self.logger.warning(f"No reward function found for strategy {strategy_name}")
            return 0.0
        reward = reward_func(state)
        self.logger.debug(f"Reward for {strategy_name} in state {state}: {reward}")
        return reward

    def update_q_table(self, state: str, action: str, reward: float, next_state: str):
        """
        Updates the Q-table using the Q-learning algorithm.

        Args:
        - state (str): The current state.
        - action (str): The action taken.
        - reward (float): Reward received after taking the action.
        - next_state (str): The resulting state after the action.
        """
        current_q = self.q_table.get((state, action), 0.0)
        max_future_q = max([self.q_table.get((next_state, a), 0.0) for a in self.strategies.keys()], default=0.0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[(state, action)] = new_q
        self.logger.info(f"Updated Q-value for state-action ({state}, {action}): {new_q}")

    def select_action(self, state: str, epsilon: float = 0.1) -> str:
        """
        Selects the next action based on an epsilon-greedy policy.

        Args:
        - state (str): The current state.
        - epsilon (float): Probability of choosing a random action (exploration rate).

        Returns:
        - str: The selected strategy name to execute.
        """
        if np.random.rand() < epsilon:
            action = np.random.choice(list(self.strategies.keys()))
            self.logger.debug(f"Exploratory action selected: {action}")
        else:
            q_values = {action: self.q_table.get((state, action), 0.0) for action in self.strategies.keys()}
            action = max(q_values, key=q_values.get)
            self.logger.debug(f"Best action selected: {action} with Q-value {q_values[action]}")
        return action

    def optimize_strategy(self, strategy_name: str, state: str, next_state: str):
        """
        Runs the Q-learning optimization process for a strategy.

        Args:
        - strategy_name (str): Name of the strategy being optimized.
        - state (str): Current state of the strategy.
        - next_state (str): Next state after executing the strategy.
        """
        reward = self.calculate_reward(strategy_name, state)
        self.update_q_table(state, strategy_name, reward, next_state)
        self.logger.info(f"Optimized strategy {strategy_name} for transition {state} -> {next_state}")
