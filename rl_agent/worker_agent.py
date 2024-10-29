# Full file path: /rl_agent/worker_agent.py

from centralized_logger import CentralizedLogger
from ai.models.trading_model import TradingModel  # Assumed model for executing trades
import random

logger = CentralizedLogger()

class WorkerAgent:
    """
    Executes trades to achieve immediate profitability and contribute to NAV targets.
    Enhanced with realistic trade decision-making and model-based adjustments.
    """

    def __init__(self, initial_balance=10000, transaction_fee=0.001):
        """
        Initializes the WorkerAgent with a trading model, balance, and transaction fee.

        Args:
            initial_balance (float): Initial balance for the agent.
            transaction_fee (float): Transaction fee percentage for each trade.
        """
        self.trading_model = TradingModel(model_type="PPO")  # Reinforcement learning model
        self.target_nav = None
        self.balance = initial_balance
        self.last_trade_profit = 0
        self.transaction_fee = transaction_fee

    def set_target_nav(self, target_nav):
        """
        Sets the target NAV for this worker’s trades.

        Args:
            target_nav (float): The target NAV this agent aims to reach.
        """
        self.target_nav = target_nav
        logger.log("info", f"WorkerAgent target NAV set to: {self.target_nav}")

    def execute_trade(self, market_data):
        """
        Executes a trade to align the current NAV with the target NAV and calculates rewards.

        Args:
            market_data (dict): Real-time market data for making trade decisions.
        """
        if self.target_nav is None:
            logger.log("warning", "No target NAV set for WorkerAgent. Skipping trade execution.")
            return

        # Predict trade action using the trading model
        trade_action = self.trading_model.predict(market_data, target_nav=self.target_nav)
        self.perform_trade(trade_action, market_data)

    def perform_trade(self, trade_action, market_data):
        """
        Executes a trade based on the action from the model, updating balance and calculating reward.

        Args:
            trade_action (dict): Model’s trade action, including type and amount.
            market_data (dict): Market data to simulate real-time volatility.
        """
        trade_amount = trade_action.get("amount", 0)
        trade_type = trade_action.get("type", "buy")
        market_volatility = market_data.get("volatility", random.uniform(0.01, 0.05))
        
        # Calculate transaction fee and adjust trade amount for volatility
        effective_trade_amount = trade_amount * (1 - self.transaction_fee)
        profit_multiplier = (1 + market_volatility) if trade_type == "buy" else (1 - market_volatility)
        
        # Calculate profit or loss based on trade type
        profit = self.calculate_trade_profit(trade_type, effective_trade_amount, profit_multiplier, trade_amount)

        # Update balance and log the trade details
        self.balance += profit
        trade_result = {"profit": profit, "trade_type": trade_type, "amount": trade_amount, "net_balance": self.balance}
        reward = self.calculate_reward(trade_result)
        logger.log("info", f"Executed {trade_type} trade: Amount: {trade_amount}, Profit: {profit}, Reward: {reward}")

    def calculate_trade_profit(self, trade_type, effective_trade_amount, profit_multiplier, trade_amount):
        """
        Determines profit or loss based on trade type and market conditions.

        Args:
            trade_type (str): Type of trade, either 'buy' or 'sell'.
            effective_trade_amount (float): Amount adjusted for transaction fees.
            profit_multiplier (float): Adjusted multiplier based on market volatility.
            trade_amount (float): Initial trade amount.

        Returns:
            float: Calculated profit or loss from the trade.
        """
        if trade_type == "buy":
            return effective_trade_amount * profit_multiplier - trade_amount
        elif trade_type == "sell":
            return trade_amount - effective_trade_amount * profit_multiplier
        else:
            logger.log("error", f"Unknown trade type: {trade_type}")
            return 0.0

    def calculate_reward(self, trade_result):
        """
        Calculates the reward based on immediate trade profit, with slight variation for realism.

        Args:
            trade_result (dict): Contains details of the trade, including 'profit'.

        Returns:
            float: Reward based on immediate trade profit, adjusted with minor random variance.
        """
        profit = trade_result.get("profit", 0)
        self.last_trade_profit = profit
        return profit * (1 + random.uniform(0.01, 0.02))  # Adds a slight variance to the reward
