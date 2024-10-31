# moneyverse/managers/transaction_manager.py

import logging
from typing import Dict, Optional
from moneyverse.database.db_connection import DatabaseConnection

class TransactionManager:
    """
    Manages trade execution, tracking, and logging across assets and exchanges.

    Attributes:
    - db (DatabaseConnection): Database connection for logging transactions.
    - logger (Logger): Logs trade executions and any failures or risks.
    """

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.logger.info("TransactionManager initialized.")

    def execute_trade(self, wallet, asset: str, amount: float, price: float, action: str, exchange: str) -> Optional[Dict[str, float]]:
        """
        Executes a trade and logs it to the database.

        Args:
        - wallet (Wallet): Wallet instance for executing the trade.
        - asset (str): Asset to trade.
        - amount (float): Quantity of the asset to trade.
        - price (float): Current price for the trade.
        - action (str): "buy" or "sell" action.
        - exchange (str): Name of the exchange where the trade is executed.

        Returns:
        - dict: Details of the executed trade if successful, None if failed.
        """
        try:
            trade_value = amount * price
            if action == "buy":
                wallet.update_balance(asset, trade_value)
                self.logger.info(f"Executed buy of {amount} {asset} at {price} on {exchange}.")
            elif action == "sell":
                wallet.update_balance(asset, -trade_value)
                self.logger.info(f"Executed sell of {amount} {asset} at {price} on {exchange}.")
            else:
                self.logger.error(f"Invalid trade action: {action}")
                return None

            # Log transaction in the database
            self.db.log_transaction(asset, amount, price, action, exchange)
            return {
                "asset": asset,
                "amount": amount,
                "price": price,
                "action": action,
                "exchange": exchange,
                "trade_value": trade_value
            }

        except Exception as e:
            self.logger.error(f"Trade execution failed for {amount} {asset} at {price} on {exchange}: {str(e)}")
            return None

    def monitor_trade(self, asset: str, target_price: float, current_price: float) -> bool:
        """
        Monitors an open trade for target price and manages risk.

        Args:
        - asset (str): Asset being monitored.
        - target_price (float): Desired target price to achieve profit.
        - current_price (float): Current market price of the asset.

        Returns:
        - bool: True if target achieved, False if monitoring continues.
        """
        if current_price >= target_price:
            self.logger.info(f"Target price of {target_price} reached for {asset}.")
            return True
        else:
            self.logger.debug(f"Target price not reached for {asset}. Current price: {current_price}")
            return False

    def log_trade_failure(self, asset: str, amount: float, action: str, reason: str):
        """
        Logs a failed trade attempt.

        Args:
        - asset (str): Asset for which the trade failed.
        - amount (float): Quantity of the asset.
        - action (str): "buy" or "sell" action.
        - reason (str): Reason for trade failure.
        """
        self.logger.warning(f"Trade failure logged for {amount} {asset}, action: {action}, reason: {reason}.")
        self.db.log_trade_failure(asset, amount, action, reason)

    def update_trade_status(self, trade_id: int, status: str):
        """
        Updates the status of a trade in the database.

        Args:
        - trade_id (int): ID of the trade to update.
        - status (str): New status, e.g., "completed" or "failed".
        """
        self.db.update_trade_status(trade_id, status)
        self.logger.info(f"Updated trade ID {trade_id} to status: {status}")
