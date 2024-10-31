# moneyverse/managers/database_manager.py

import logging
from moneyverse.database.db_connection import DatabaseConnection

class DatabaseManager:
    """
    Manages all database operations, including transaction logging, balance tracking, and error logging.

    Attributes:
    - db_connection (DatabaseConnection): Connection to the database.
    - logger (Logger): Logs database actions and errors.
    """

    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        self.logger = logging.getLogger(__name__)
        self.logger.info("DatabaseManager initialized with database connection.")

    def log_transaction(self, transaction_data: dict):
        """
        Logs a completed transaction in the database.

        Args:
        - transaction_data (dict): Data containing transaction details such as asset, amount, price, and action.
        """
        try:
            self.db_connection.insert_transaction(transaction_data)
            self.logger.info(f"Logged transaction: {transaction_data}")
        except Exception as e:
            self.logger.error(f"Failed to log transaction {transaction_data}: {str(e)}")

    def update_wallet_balance(self, wallet_address: str, asset: str, balance: float):
        """
        Updates the balance of a specific asset in a wallet.

        Args:
        - wallet_address (str): Address of the wallet.
        - asset (str): Type of asset to update.
        - balance (float): New balance of the asset.
        """
        try:
            self.db_connection.update_balance(wallet_address, asset, balance)
            self.logger.info(f"Updated balance for {wallet_address}: {asset} = {balance}")
        except Exception as e:
            self.logger.error(f"Failed to update balance for {wallet_address}, asset {asset}: {str(e)}")

    def log_error(self, error_message: str):
        """
        Logs an error in the error log table for future debugging.

        Args:
        - error_message (str): Detailed error message to log.
        """
        try:
            self.db_connection.insert_error_log(error_message)
            self.logger.warning(f"Logged error: {error_message}")
        except Exception as e:
            self.logger.error(f"Failed to log error '{error_message}': {str(e)}")

    def get_transaction_history(self, wallet_address: str) -> list:
        """
        Retrieves the transaction history for a specific wallet.

        Args:
        - wallet_address (str): Address of the wallet.

        Returns:
        - list: List of transactions associated with the wallet.
        """
        try:
            history = self.db_connection.fetch_transaction_history(wallet_address)
            self.logger.info(f"Retrieved transaction history for {wallet_address}.")
            return history
        except Exception as e:
            self.logger.error(f"Failed to retrieve transaction history for {wallet_address}: {str(e)}")
            return []

    def get_wallet_balances(self) -> dict:
        """
        Retrieves the balances of all wallets in the system.

        Returns:
        - dict: Dictionary of wallet addresses and their asset balances.
        """
        try:
            balances = self.db_connection.fetch_all_balances()
            self.logger.info("Retrieved wallet balances.")
            return balances
        except Exception as e:
            self.logger.error("Failed to retrieve wallet balances: " + str(e))
            return {}

    def backup_database(self, backup_path: str):
        """
        Creates a backup of the current database.

        Args:
        - backup_path (str): File path to store the database backup.
        """
        try:
            self.db_connection.create_backup(backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
        except Exception as e:
            self.logger.error(f"Failed to backup database to {backup_path}: {str(e)}")
