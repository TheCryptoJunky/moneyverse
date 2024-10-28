# Full file path: /moneyverse/managers/transaction_manager.py

import requests
from utils.retry_decorator import retry
from centralized_logger import CentralizedLogger

logger = CentralizedLogger()

class TransactionManager:
    """
    Manages transactions, including retry logic for resilience.
    """

    def __init__(self):
        self.transactions = []

    @retry(retries=5, delay=2, backoff=1.5)
    def fetch_data_from_api(self, api_url):
        """
        Fetches data from an external API with retry and fallback.

        Parameters:
            api_url (str): The API endpoint to call.

        Returns:
            dict: Response data if successful.
        """
        response = requests.get(api_url)
        response.raise_for_status()  # Raises HTTPError for bad responses
        logger.info(f"Data successfully fetched from {api_url}")
        return response.json()

    @retry(retries=3, delay=1, backoff=2, fallback_function=lambda: {"status": "fallback"})
    def update_transaction(self, transaction_id, data):
        """
        Update transaction data with a retry mechanism for resiliency.
        Includes a fallback that logs failed transactions as 'pending'.

        Parameters:
            transaction_id (int): ID of the transaction.
            data (dict): Transaction data to update.

        Returns:
            dict: API response if successful; 'fallback' status otherwise.
        """
        try:
            response = requests.post(f"https://example.com/api/update/{transaction_id}", json=data)
            response.raise_for_status()
            logger.info(f"Transaction {transaction_id} updated successfully.")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update transaction {transaction_id}: {e}")
            raise
