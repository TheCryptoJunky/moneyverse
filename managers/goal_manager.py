# moneyverse/managers/goal_manager.py

import logging
from typing import Dict

class GoalManager:
    """
    Manages performance goals for individual strategies and the overall portfolio.

    Attributes:
    - strategy_goals (dict): Stores performance targets for individual strategies.
    - portfolio_goal (float): Net Asset Value (NAV) growth target for the entire portfolio.
    - logger (Logger): Logs goal updates, progress tracking, and adjustments.
    """

    def __init__(self, initial_portfolio_goal=0.05):
        self.strategy_goals = {}  # {strategy_name: goal}
        self.portfolio_goal = initial_portfolio_goal  # e.g., 5% growth target
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"GoalManager initialized with portfolio goal: {self.portfolio_goal * 100}%")

    def set_strategy_goal(self, strategy_name: str, goal: float):
        """
        Sets a performance target for an individual strategy.

        Args:
        - strategy_name (str): Name of the strategy.
        - goal (float): Target performance as a percentage (e.g., 0.1 for 10% growth).
        """
        self.strategy_goals[strategy_name] = goal
        self.logger.info(f"Set goal for {strategy_name}: {goal * 100}%")

    def update_portfolio_goal(self, new_goal: float):
        """
        Updates the growth target for the portfolio.

        Args:
        - new_goal (float): New target growth percentage (e.g., 0.07 for 7%).
        """
        self.portfolio_goal = new_goal
        self.logger.info(f"Updated portfolio goal to {new_goal * 100}%")

    def check_strategy_progress(self, strategy_name: str, performance: float) -> bool:
        """
        Checks if a strategy has met its performance target.

        Args:
        - strategy_name (str): Name of the strategy.
        - performance (float): Current performance percentage.

        Returns:
        - bool: True if the goal is met, False otherwise.
        """
        target = self.strategy_goals.get(strategy_name, None)
        if target is not None:
            if performance >= target:
                self.logger.info(f"{strategy_name} met its goal with performance of {performance * 100}%")
                return True
            else:
                self.logger.debug(f"{strategy_name} progress: {performance * 100}% of target {target * 100}%")
        return False

    def check_portfolio_progress(self, current_nav: float, initial_nav: float) -> bool:
        """
        Checks if the portfolio has met its growth target based on NAV.

        Args:
        - current_nav (float): Current NAV of the portfolio.
        - initial_nav (float): Initial NAV at the start of the goal period.

        Returns:
        - bool: True if the goal is met, False otherwise.
        """
        growth = (current_nav - initial_nav) / initial_nav
        if growth >= self.portfolio_goal:
            self.logger.info(f"Portfolio goal met with growth of {growth * 100}%")
            return True
        else:
            self.logger.debug(f"Portfolio growth: {growth * 100}% of target {self.portfolio_goal * 100}%")
        return False

    def adjust_goals_based_on_market(self, market_volatility: float):
        """
        Adjusts goals based on market conditions, lowering targets in volatile conditions.

        Args:
        - market_volatility (float): Current volatility measure (e.g., VIX index or custom volatility metric).
        """
        if market_volatility > 0.5:  # High volatility threshold
            self.portfolio_goal *= 0.9  # Lower goals by 10% in high volatility
            self.strategy_goals = {strategy: goal * 0.9 for strategy, goal in self.strategy_goals.items()}
            self.logger.info("Goals adjusted for high market volatility.")
        else:
            self.logger.info("No adjustment needed for stable market conditions.")

    def reset_strategy_goals(self):
        """
        Resets all strategy goals to the initial baseline, allowing fresh tracking.
        """
        for strategy in self.strategy_goals:
            self.strategy_goals[strategy] = self.strategy_goals.get(strategy, self.portfolio_goal)
        self.logger.info("Reset all strategy goals to baseline targets.")
