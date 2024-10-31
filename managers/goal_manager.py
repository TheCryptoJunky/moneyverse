# moneyverse/managers/goal_manager.py

import logging
from typing import Dict

class GoalManager:
    """
    Manages adaptive trading goals such as profit targets, risk limits, and trade frequency.

    Attributes:
    - profit_target (float): Desired profit target as a percentage.
    - risk_limit (float): Maximum acceptable risk per trade as a percentage.
    - trade_frequency (int): Desired frequency of trades within a given period.
    - logger (Logger): Logs goal setting and updates.
    """

    def __init__(self, profit_target=0.05, risk_limit=0.02, trade_frequency=10):
        self.profit_target = profit_target  # 5% profit target
        self.risk_limit = risk_limit  # 2% risk per trade limit
        self.trade_frequency = trade_frequency  # Default trade frequency
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"GoalManager initialized with profit target: {self.profit_target * 100}%, "
                         f"risk limit: {self.risk_limit * 100}%, trade frequency: {self.trade_frequency}")

    def update_goal(self, goal_name: str, value: float):
        """
        Updates a specified trading goal.

        Args:
        - goal_name (str): Name of the goal to update ("profit_target", "risk_limit", or "trade_frequency").
        - value (float): New value for the specified goal.
        """
        if goal_name == "profit_target":
            self.profit_target = value
            self.logger.info(f"Updated profit target to {self.profit_target * 100}%")
        elif goal_name == "risk_limit":
            self.risk_limit = value
            self.logger.info(f"Updated risk limit to {self.risk_limit * 100}%")
        elif goal_name == "trade_frequency":
            self.trade_frequency = int(value)
            self.logger.info(f"Updated trade frequency to {self.trade_frequency}")
        else:
            self.logger.warning(f"Goal '{goal_name}' not recognized. No update performed.")

    def evaluate_performance(self, current_profit: float) -> bool:
        """
        Evaluates current profit against the profit target to determine if adjustments are needed.

        Args:
        - current_profit (float): Current profit as a percentage.

        Returns:
        - bool: True if profit target is met or exceeded, False otherwise.
        """
        if current_profit >= self.profit_target:
            self.logger.info(f"Profit target of {self.profit_target * 100}% met. Current profit: {current_profit * 100}%.")
            return True
        self.logger.debug("Profit target not yet met.")
        return False

    def risk_exceeded(self, current_risk: float) -> bool:
        """
        Checks if the current risk exceeds the defined risk limit.

        Args:
        - current_risk (float): Current risk level as a percentage.

        Returns:
        - bool: True if risk limit is exceeded, False otherwise.
        """
        if current_risk > self.risk_limit:
            self.logger.warning(f"Risk limit exceeded. Current risk: {current_risk * 100}%, limit: {self.risk_limit * 100}%.")
            return True
        self.logger.debug("Risk limit within acceptable bounds.")
        return False

    def adjust_trade_frequency(self, market_condition: str):
        """
        Adjusts trade frequency based on market conditions.

        Args:
        - market_condition (str): Describes current market condition (e.g., "volatile", "stable").
        """
        if market_condition == "volatile":
            self.trade_frequency = max(1, int(self.trade_frequency * 0.5))  # Reduce frequency
            self.logger.info(f"Market is volatile. Reducing trade frequency to {self.trade_frequency}.")
        elif market_condition == "stable":
            self.trade_frequency = int(self.trade_frequency * 1.2)  # Increase frequency
            self.logger.info(f"Market is stable. Increasing trade frequency to {self.trade_frequency}.")

    def get_goals(self) -> Dict[str, float]:
        """
        Retrieves the current set of goals.

        Returns:
        - dict: Dictionary containing current goals.
        """
        return {
            "profit_target": self.profit_target,
            "risk_limit": self.risk_limit,
            "trade_frequency": self.trade_frequency
        }
