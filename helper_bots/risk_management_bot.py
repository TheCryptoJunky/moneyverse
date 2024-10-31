# moneyverse/helper_bots/risk_management_bot.py

import logging
from typing import Dict

class RiskManagementBot:
    """
    Monitors and manages risk exposure across strategies, adapting risk parameters based on real-time metrics.

    Attributes:
    - max_exposure (float): Maximum exposure allowed per strategy.
    - max_daily_loss (float): Maximum allowable daily loss.
    - max_drawdown (float): Maximum allowable drawdown.
    - current_exposure (dict): Tracks current exposure for each strategy.
    - logger (Logger): Logs risk adjustments and alerts.
    """

    def __init__(self, max_exposure=0.05, max_daily_loss=0.1, max_drawdown=0.2):
        self.max_exposure = max_exposure  # Default 5% max exposure per strategy
        self.max_daily_loss = max_daily_loss  # Default 10% daily loss cap
        self.max_drawdown = max_drawdown  # Default 20% drawdown limit
        self.current_exposure = {}  # {strategy_name: exposure}
        self.daily_losses = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("RiskManagementBot initialized with base limits.")

    def set_strategy_limits(self, strategy: str, exposure_limit: float, daily_loss_limit: float):
        """
        Sets risk parameters for a specific strategy.

        Args:
        - strategy (str): Name of the strategy.
        - exposure_limit (float): Maximum exposure for the strategy.
        - daily_loss_limit (float): Maximum daily loss allowed for the strategy.
        """
        self.current_exposure[strategy] = 0  # Initialize current exposure
        self.daily_losses[strategy] = 0  # Initialize daily loss tracker
        self.max_exposure = exposure_limit
        self.max_daily_loss = daily_loss_limit
        self.logger.info(f"Set limits for {strategy}: Exposure {exposure_limit}, Daily Loss {daily_loss_limit}")

    def update_exposure(self, strategy: str, amount: float):
        """
        Updates the exposure for a strategy based on a new trade.

        Args:
        - strategy (str): Name of the strategy.
        - amount (float): Value of the exposure added or reduced.
        """
        if strategy in self.current_exposure:
            self.current_exposure[strategy] += amount
            self.logger.debug(f"Updated exposure for {strategy}: {self.current_exposure[strategy]}")
        else:
            self.logger.error(f"Strategy {strategy} not initialized in risk manager.")

    def check_exposure(self, strategy: str) -> bool:
        """
        Checks if the current exposure for a strategy exceeds the maximum allowable exposure.

        Args:
        - strategy (str): Name of the strategy.

        Returns:
        - bool: True if exposure is within limit, False if exceeded.
        """
        exposure = self.current_exposure.get(strategy, 0)
        if exposure > self.max_exposure:
            self.logger.warning(f"Exposure for {strategy} exceeds limit: {exposure} > {self.max_exposure}")
            return False
        return True

    def update_daily_loss(self, strategy: str, loss_amount: float):
        """
        Tracks and updates the daily loss for a strategy.

        Args:
        - strategy (str): Name of the strategy.
        - loss_amount (float): Amount of loss to record.
        """
        if strategy in self.daily_losses:
            self.daily_losses[strategy] += loss_amount
            self.logger.debug(f"Updated daily loss for {strategy}: {self.daily_losses[strategy]}")
        else:
            self.logger.error(f"Strategy {strategy} not initialized in risk manager.")

    def check_daily_loss(self, strategy: str) -> bool:
        """
        Checks if the daily loss for a strategy exceeds the maximum allowable daily loss.

        Args:
        - strategy (str): Name of the strategy.

        Returns:
        - bool: True if daily loss is within limit, False if exceeded.
        """
        daily_loss = self.daily_losses.get(strategy, 0)
        if daily_loss > self.max_daily_loss:
            self.logger.warning(f"Daily loss for {strategy} exceeds limit: {daily_loss} > {self.max_daily_loss}")
            return False
        return True

    def reset_daily_losses(self):
        """
        Resets daily losses at the end of each trading day.
        """
        self.daily_losses = {strategy: 0 for strategy in self.daily_losses}
        self.logger.info("Daily losses reset for all strategies.")

    def adjust_risk_parameters(self, market_volatility: float):
        """
        Dynamically adjusts risk parameters based on market volatility.

        Args:
        - market_volatility (float): Current market volatility measure.
        """
        if market_volatility > 0.5:  # Threshold for high volatility
            # Increase limits conservatively during volatile times
            self.max_exposure *= 0.8
            self.max_daily_loss *= 0.8
            self.max_drawdown *= 0.9
            self.logger.info("Adjusted risk parameters for high volatility.")
        else:
            # Reset to base limits in stable markets
            self.max_exposure = 0.05
            self.max_daily_loss = 0.1
            self.max_drawdown = 0.2
            self.logger.info("Risk parameters reset for stable market conditions.")
