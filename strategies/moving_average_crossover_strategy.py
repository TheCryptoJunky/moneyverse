# moneyverse/strategies/moving_average_crossover_strategy.py

import logging
from typing import List

class MovingAverageCrossoverStrategy:
    """
    Detects buying and selling signals based on the crossover of short-term and long-term moving averages.

    Attributes:
    - short_window (int): Period for the short-term moving average.
    - long_window (int): Period for the long-term moving average.
    - logger (Logger): Logs bot actions and detected opportunities.
    """

    def __init__(self, short_window=10, long_window=50):
        self.short_window = short_window  # Short-term moving average window
        self.long_window = long_window  # Long-term moving average window
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MovingAverageCrossoverStrategy initialized with short window: {self.short_window}, long window: {self.long_window}")

    def calculate_moving_average(self, prices: List[float], window: int) -> float:
        """
        Calculates the moving average for a given price history and window size.

        Args:
        - prices (list): List of historical prices.
        - window (int): Number of data points to include in moving average calculation.

        Returns:
        - float: Calculated moving average of prices.
        """
        if len(prices) < window:
            self.logger.warning(f"Not enough data to calculate {window}-period moving average.")
            return sum(prices) / len(prices)  # Fallback if less data available
        moving_average = sum(prices[-window:]) / window
        self.logger.debug(f"Calculated {window}-period moving average: {moving_average}")
        return moving_average

    def detect_crossover(self, short_ma: float, long_ma: float) -> str:
        """
        Determines if there is a crossover between the short and long moving averages.

        Args:
        - short_ma (float): Short-term moving average.
        - long_ma (float): Long-term moving average.

        Returns:
        - str: "buy" if short crosses above long, "sell" if short crosses below long, otherwise empty.
        """
        if short_ma > long_ma:
            self.logger.info("Bullish crossover detected: Short MA above Long MA - Buy signal.")
            return "buy"
        elif short_ma < long_ma:
            self.logger.info("Bearish crossover detected: Short MA below Long MA - Sell signal.")
            return "sell"
        self.logger.debug("No crossover detected.")
        return ""

    def execute_trade(self, wallet, action: str, amount: float, price: float):
        """
        Executes a trade based on the crossover signal.

        Args:
        - wallet (Wallet): Wallet instance to execute the trade.
        - action (str): "buy" or "sell" action.
        - amount (float): Amount of asset to trade.
        - price (float): Current price at which to trade.
        """
        if action == "buy":
            wallet.update_balance("buy", amount * price)
            self.logger.info(f"Executed crossover buy trade for {amount} at {price}.")
        elif action == "sell":
            wallet.update_balance("sell", -amount * price)
            self.logger.info(f"Executed crossover sell trade for {amount} at {price}.")

    def run(self, wallet, price_history: List[float], current_price: float, amount: float):
        """
        Detects and executes trades based on moving average crossovers.

        Args:
        - wallet (Wallet): Wallet instance for executing trades.
        - price_history (list): Historical price data for calculating moving averages.
        - current_price (float): Current market price of the asset.
        - amount (float): Amount to trade if an opportunity is detected.
        """
        short_ma = self.calculate_moving_average(price_history, self.short_window)
        long_ma = self.calculate_moving_average(price_history, self.long_window)
        action = self.detect_crossover(short_ma, long_ma)
        if action:
            self.execute_trade(wallet, action, amount, current_price)
        else:
            self.logger.info("No moving average crossover trade executed; no suitable opportunity detected.")
