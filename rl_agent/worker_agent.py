# Full file path: /rl_agent/worker_agent.py

from centralized_logger import CentralizedLogger
from ai.models.trading_model import TradingModel  # Assumed model for executing trades
import random

logger = CentralizedLogger()

class WorkerAgent:
    """
    Executes trades based on ManagerAgent's directives, focusing on achieving immediate profitability
    and contributing towards NAV targets. Enhanced with complex model adjustments for realistic trading behavior.
    """

    def __init__(self, initial_balance=10000, transaction_fee=0.001):
        self.trading_model = TradingModel(model_type="PPO")  # Reinforcement learning model
        self.target_nav = None
        self.balance = initial_balance
        self.last_trade_profit = 0
        self.transaction_fee = transaction_fee  # Fixed fee for each transaction as a percentage

    def set_target_nav(self, target_nav):
        """Sets the target NAV for this worker's trades."""
        self.target_nav = target_nav
        logger.log("info", f"WorkerAgent target NAV set to: {self.target_nav}")

    def execute_trade(self, market_data):
        """
        Executes a trade to move closer to the target NAV and calculates reward based on trade profitability.
        
        Args:
            market_data (dict): Real-time market data for trade decision-making.
        """
        if self.target_nav is None:
            logger.log("warning", "No target NAV set for WorkerAgent. Skipping trade execution.")
            return

        # Make trade decision based on target NAV
        trade_action = self.trading_model.predict(market_data, target_nav=self.target_nav)
        self.perform_trade(trade_action, market_data)

    def perform_trade(self, trade_action, market_data):
        """
        Executes the actual trade by adjusting balance based on trade action and calculating reward.
        
        Args:
            trade_action (dict): Details of the trade action from the model.
            market_data (dict): Real-time market data to simulate price volatility.
        """
        trade_amount = trade_action.get("amount", 0)
        trade_type = trade_action.get("type", "buy")
        market_volatility = market_data.get("volatility", random.uniform(0.01, 0.05))
        
        # Calculate transaction fee and adjust trade amount based on volatility
        effective_trade_amount = trade_amount * (1 - self.transaction_fee)
        profit_multiplier = (1 + market_volatility) if trade_type == "buy" else (1 - market_volatility)
        
        # Determine profit or loss based on trade type and market conditions
        if trade_type == "buy":
            profit = effective_trade_amount * profit_multiplier - trade_amount
            self.balance += effective_trade_amount + profit
        elif trade_type == "sell":
            profit = trade_amount - effective_trade_amount * profit_multiplier
            self.balance += profit
        else:
            logger.log("error", f"Unknown trade type: {trade_type}")
            return

        # Log trade details and calculate reward
        trade_result = {"profit": profit, "trade_type": trade_type, "amount": trade_amount, "net_amount": self.balance}
        reward = self.calculate_reward(trade_result)
        logger.log("info", f"Executed {trade_type} trade: Amount: {trade_amount}, Profit: {profit}, Reward: {reward}")

    def calculate_reward(self, trade_result):
        """
        Calculates reward based on the immediate profit of the trade, adjusted by market conditions.
        
        Args:
            trade_result (dict): Contains details of the trade, including 'profit'.
        
        Returns:
            float: Reward based on immediate trade profit.
        """
        profit = trade_result.get("profit", 0)
        self.last_trade_profit = profit
        return profit * (1 + random.uniform(0.01, 0.02))  # Adjust reward slightly for realistic variance
