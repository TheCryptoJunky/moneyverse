# moneyverse/strategies/mean_reversion_strategy.py

import logging
from typing import Dict

class MeanReversionStrategy:
    """
    Detects mean reversion opportunities by identifying when assets deviate significantly from their average price.

    Attributes:
    - threshold (float): Minimum deviation from the mean to trigger a trade.
    - lookback_period (int): Number of data points to calculate the moving average.
    - logger (Logger): Logs bot actions and detected opportunities.
    """

    def __init__(self, threshold=0.02, lookback_period=20):
        self.threshold = threshold  # Minimum deviation to trigger trades
        self.lookback_period = lookback_period  # Period to calculate the moving average
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MeanReversionStrategy initialized with threshold: {self.threshold * 100}%, lookback period: {self.lookback_period}")

    def calculate_moving_average(self, price_history: list) -> float:
        """
        Calculates the moving average for a given price history.

        Args:
        - price_history (list): List of historical prices.

        Returns:
        - float: Calculated moving average of prices.
        """
        if len(price_history) < self.lookback_period:
            self.logger.warning("Not enough data to calculate moving average.")
            return sum(price_history) / len(price_history)  # Fallback if less data available
        moving_average = sum(price_history[-self.lookback_period:]) / self.lookback_period
        self.logger.debug(f"Calculated moving average: {moving_average}")
        return moving_average

    def detect_opportunity(self, current_price: float, moving_average: float) -> str:
        """
        Determines if there is a mean reversion trading opportunity.

        Args:
        - current_price (float): Current price of the asset.
        - moving_average (float): Calculated moving average price.

        Returns:
        - str: "buy" if current price is below threshold, "sell" if above threshold, or empty if no action.
        """
        deviation = (current_price - moving_average) / moving_average

        if deviation < -self.threshold:
            self.logger.info(f"Mean reversion buy opportunity detected. Deviation: {deviation:.2%}")
            return "buy"
        elif deviation > self.threshold:
            self.logger.info(f"Mean reversion sell opportunity detected. Deviation: {deviation:.2%}")
            return "sell"
        self.logger.debug("No mean reversion opportunity detected.")
        return ""

    def execute_trade(self, wallet, action: str, amount: float, price: float):
        """
        Executes a trade based on the mean reversion opportunity detected.

        Args:
        - wallet (Wallet): Wallet instance to execute the trade.
        - action (str): "buy" or "sell" action.
        - amount (float): Amount of asset to trade.
        - price (float): Current price at which to trade.
        """
        if action == "buy":
            wallet.update_balance("buy", amount * price)
            self.logger.info(f"Executed mean reversion buy trade for {amount} at {price}.")
        elif action == "sell":
            wallet.update_balance("sell", -amount * price)
            self.logger.info(f"Executed mean reversion sell trade for {amount} at {price}.")

    def run(self, wallet, price_history: list, current_price: float, amount: float):
        """
        Detects and executes mean reversion trade based on current price and historical data.

        Args:
        - wallet (Wallet): Wallet instance to execute trades.
        - price_history (list): Historical price data for calculating mean.
        - current_price (float): Current market price of the asset.
        - amount (float): Amount to trade if an opportunity is detected.
        """
        moving_average = self.calculate_moving_average(price_history)
        action = self.detect_opportunity(current_price, moving_average)
        if action:
            self.execute_trade(wallet, action, amount, current_price)
        else:
            self.logger.info("No mean reversion trade executed; no suitable opportunity detected.")
