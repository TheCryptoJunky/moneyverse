# moneyverse/managers/risk_manager.py

import logging
from typing import Dict

class RiskManager:
    """
    Monitors risk on a per-trade and portfolio basis, enforcing limits to prevent extreme losses.

    Attributes:
    - max_risk_per_trade (float): Maximum risk allowed per trade as a percentage of the portfolio.
    - max_portfolio_drawdown (float): Maximum allowable portfolio drawdown as a percentage.
    - max_daily_loss (float): Maximum allowable daily loss as a percentage of the portfolio.
    - current_drawdown (float): Current portfolio drawdown.
    - logger (Logger): Logs risk monitoring actions and enforcements.
    """

    def __init__(self, max_risk_per_trade=0.02, max_portfolio_drawdown=0.10, max_daily_loss=0.05):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_portfolio_drawdown = max_portfolio_drawdown  # 10% drawdown limit
        self.max_daily_loss = max_daily_loss  # 5% daily loss limit
        self.current_drawdown = 0.0
        self.daily_loss = 0.0
        self.logger = logging.getLogger(__name__)
        self.logger.info("RiskManager initialized with risk limits.")

    def calculate_trade_risk(self, trade_value: float, portfolio_value: float) -> float:
        """
        Calculates the risk for a given trade as a percentage of the portfolio.

        Args:
        - trade_value (float): Value of the trade.
        - portfolio_value (float): Total portfolio value.

        Returns:
        - float: Risk percentage of the trade.
        """
        trade_risk = trade_value / portfolio_value
        self.logger.debug(f"Calculated trade risk: {trade_risk:.2%}")
        return trade_risk

    def check_trade_risk(self, trade_value: float, portfolio_value: float) -> bool:
        """
        Checks if a trade's risk exceeds the maximum allowable per trade.

        Args:
        - trade_value (float): Value of the trade.
        - portfolio_value (float): Total portfolio value.

        Returns:
        - bool: True if trade risk is within limits, False otherwise.
        """
        trade_risk = self.calculate_trade_risk(trade_value, portfolio_value)
        if trade_risk > self.max_risk_per_trade:
            self.logger.warning(f"Trade risk exceeds limit: {trade_risk:.2%} > {self.max_risk_per_trade:.2%}")
            return False
        return True

    def update_drawdown(self, portfolio_value: float, peak_value: float):
        """
        Updates the portfolio's current drawdown.

        Args:
        - portfolio_value (float): Current portfolio value.
        - peak_value (float): Peak portfolio value before the drawdown.
        """
        self.current_drawdown = (peak_value - portfolio_value) / peak_value
        if self.current_drawdown > self.max_portfolio_drawdown:
            self.logger.warning(f"Portfolio drawdown exceeds limit: {self.current_drawdown:.2%} > {self.max_portfolio_drawdown:.2%}")
        else:
            self.logger.debug(f"Current drawdown: {self.current_drawdown:.2%}")

    def check_daily_loss_limit(self, daily_loss: float, portfolio_value: float) -> bool:
        """
        Checks if the daily loss exceeds the maximum allowable loss.

        Args:
        - daily_loss (float): Current daily loss.
        - portfolio_value (float): Total portfolio value.

        Returns:
        - bool: True if daily loss is within limits, False otherwise.
        """
        loss_percentage = daily_loss / portfolio_value
        if loss_percentage > self.max_daily_loss:
            self.logger.warning(f"Daily loss limit exceeded: {loss_percentage:.2%} > {self.max_daily_loss:.2%}")
            return False
        return True

    def enforce_risk_limits(self, wallet, portfolio_value: float, peak_value: float, daily_loss: float):
        """
        Enforces risk limits by monitoring drawdown and daily loss limits.

        Args:
        - wallet (Wallet): Wallet instance to manage asset liquidation if needed.
        - portfolio_value (float): Current portfolio value.
        - peak_value (float): Peak portfolio value for drawdown calculations.
        - daily_loss (float): Current daily loss.
        """
        self.update_drawdown(portfolio_value, peak_value)
        if not self.check_daily_loss_limit(daily_loss, portfolio_value):
            self.logger.info("Daily loss limit exceeded. Pausing trades.")
            # Implement logic to pause trading here.

        if self.current_drawdown > self.max_portfolio_drawdown:
            self.logger.info("Drawdown limit exceeded. Triggering partial liquidation.")
            # Implement logic for partial liquidation if drawdown exceeds limit.
