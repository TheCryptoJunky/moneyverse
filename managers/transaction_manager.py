# moneyverse/managers/transaction_manager.py

import logging
import time
from typing import Dict, Optional

class TransactionManager:
    """
    Manages the lifecycle of each transaction, ensuring retries, logging, and handling failures.

    Attributes:
    - max_retries (int): Maximum number of retries for a transaction.
    - retry_interval (float): Time in seconds between retries.
    - transaction_log (dict): Stores transaction details and statuses.
    - logger (Logger): Logs transaction statuses and issues.
    """

    def __init__(self, max_retries=3, retry_interval=5.0):
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.transaction_log = {}  # {transaction_id: {status, retries}}
        self.logger = logging.getLogger(__name__)
        self.logger.info("TransactionManager initialized with retry policies.")

    def initiate_transaction(self, transaction_id: str, transaction_data: Dict[str, any]):
        """
        Initiates a new transaction, logging it and setting its initial status.

        Args:
        - transaction_id (str): Unique identifier for the transaction.
        - transaction_data (dict): Data for the transaction (e.g., asset, amount, action).
        """
        self.transaction_log[transaction_id] = {
            "data": transaction_data,
            "status": "initiated",
            "retries": 0
        }
        self.logger.info(f"Initiated transaction {transaction_id}: {transaction_data}")

    def process_transaction(self, transaction_id: str, execute_func: callable) -> bool:
        """
        Processes a transaction by attempting to execute it with retries.

        Args:
        - transaction_id (str): Unique identifier for the transaction.
        - execute_func (callable): Function to execute the transaction.

        Returns:
        - bool: True if transaction succeeds, False if it fails after retries.
        """
        transaction = self.transaction_log.get(transaction_id)
        if not transaction:
            self.logger.error(f"Transaction {transaction_id} not found in log.")
            return False

        success = False
        while transaction["retries"] < self.max_retries:
            try:
                success = execute_func(transaction["data"])
                if success:
                    transaction["status"] = "completed"
                    self.logger.info(f"Transaction {transaction_id} completed successfully.")
                    break
                else:
                    raise Exception("Transaction execution returned False.")
            except Exception as e:
                transaction["retries"] += 1
                self.logger.warning(f"Retrying transaction {transaction_id} (Attempt {transaction['retries']}): {str(e)}")
                time.sleep(self.retry_interval)

        if not success:
            transaction["status"] = "failed"
            self.logger.error(f"Transaction {transaction_id} failed after {self.max_retries} retries.")
        return success

    def get_transaction_status(self, transaction_id: str) -> Optional[str]:
        """
        Retrieves the status of a specific transaction.

        Args:
        - transaction_id (str): Unique identifier for the transaction.

        Returns:
        - str: Current status of the transaction or None if not found.
        """
        transaction = self.transaction_log.get(transaction_id)
        if transaction:
            return transaction["status"]
        self.logger.warning(f"Transaction {transaction_id} not found.")
        return None

    def handle_failed_transactions(self):
        """
        Handles all failed transactions, potentially logging them for review or re-initiating.
        """
        for transaction_id, transaction in self.transaction_log.items():
            if transaction["status"] == "failed":
                self.logger.error(f"Handling failed transaction {transaction_id}. Review required.")

    def cancel_transaction(self, transaction_id: str):
        """
        Cancels a transaction if it has not been completed, marking it as canceled.

        Args:
        - transaction_id (str): Unique identifier for the transaction.
        """
        transaction = self.transaction_log.get(transaction_id)
        if transaction and transaction["status"] != "completed":
            transaction["status"] = "canceled"
            self.logger.info(f"Transaction {transaction_id} has been canceled.")
        elif transaction and transaction["status"] == "completed":
            self.logger.warning(f"Transaction {transaction_id} already completed and cannot be canceled.")
        else:
            self.logger.warning(f"Transaction {transaction_id} not found in log.")
