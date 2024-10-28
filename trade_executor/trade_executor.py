# moneyverse/trade_executor.py

import time

class TradeExecutor:
    def __init__(self, timeout):
        self.timeout = timeout

    def execute_trade(self, trade):
        try:
            # Simulate trade execution
            time.sleep(self.timeout)
            print(f'Trade executed: {trade}')
        except Exception as e:
            print(f'Error executing trade: {e}')

# trade_executor.py

import logging
from typing import List, Dict

from src.rl_agent.wallet import Wallet
from trade import Trade
from src.rl_agent.wallet_manager import WalletManager

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, wallet_manager: WalletManager):
        self.wallet_manager = wallet_manager

    def execute_trade(self, trade: Trade) -> bool:
        """
        Execute a trade using the provided trade object.

        Args:
        trade (Trade): The trade object containing the trade details.

        Returns:
        bool: True if the trade was executed successfully, False otherwise.
        """
        # Get the source and target wallets for the trade
        source_wallet = self.wallet_manager.get_wallet(trade.source_asset)
        target_wallet = self.wallet_manager.get_wallet(trade.target_asset)

        # Check if the source wallet has sufficient balance
        if source_wallet.balance < trade.amount:
            logger.error(f"Insufficient balance in source wallet {source_wallet.asset}")
            return False

        # Check if the target wallet is available
        if target_wallet is None:
            logger.error(f"Target wallet {trade.target_asset} not found")
            return False

        # Execute the trade
        try:
            # Subtract the trade amount from the source wallet
            source_wallet.balance -= trade.amount

            # Add the trade amount to the target wallet
            target_wallet.balance += trade.amount

            # Update the wallet balances in the wallet manager
            self.wallet_manager.update_wallet_balances()

            logger.info(f"Trade executed successfully: {trade}")
            return True
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def get_available_trades(self) -> List[Trade]:
        """
        Get a list of available trades based on the current wallet balances.

        Returns:
        List[Trade]: A list of available trades.
        """
        available_trades = []

        # Iterate through each wallet in the wallet manager
        for wallet in self.wallet_manager.wallets:
            # Check if the wallet has a balance greater than 0
            if wallet.balance > 0:
                # Create a trade object for each possible trade
                for target_asset in self.wallet_manager.get_assets():
                    if target_asset != wallet.asset:
                        trade = Trade(wallet.asset, target_asset, wallet.balance)
                        available_trades.append(trade)

        return available_trades

    def get_trade_history(self) -> List[Dict]:
        """
        Get a list of previous trades.

        Returns:
        List[Dict]: A list of previous trades, where each trade is represented as a dictionary.
        """
        trade_history = []

        # Iterate through each wallet in the wallet manager
        for wallet in self.wallet_manager.wallets:
            # Get the trade history for each wallet
            wallet_trade_history = wallet.get_trade_history()

            # Add the wallet trade history to the overall trade history
            trade_history.extend(wallet_trade_history)

        return trade_history
