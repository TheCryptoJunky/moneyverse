# mocks.py
import logging
import random

def mock_wallet_balance(wallet_id):
    """
    Simulate wallet balance retrieval.

    Args:
        wallet_id (str): Wallet ID.

    Returns:
        float: Wallet balance.
    """
    return random.uniform(0, 1000)

def mock_wallet_transaction(wallet_id, amount):
    """
    Simulate wallet transaction.

    Args:
        wallet_id (str): Wallet ID.
        amount (float): Transaction amount.

    Returns:
        bool: Transaction success.
    """
    return True

def mock_api_call(endpoint, params):
    """
    Simulate API call.

    Args:
        endpoint (str): API endpoint.
        params (dict): API parameters.

    Returns:
        dict: API response.
    """
    return {'success': True, 'data': {}}

class MockWallet:
    def __init__(self, config):
        self.config = config
        self.wallet_id = config.get('wallet_id')

    def deposit(self, amount):
        if mock_wallet_transaction(self.wallet_id, amount):
            logging.info(f"Deposited {amount} into wallet {self.wallet_id}")
        else:
            logging.error(f"Failed to deposit {amount} into wallet {self.wallet_id}")

    def withdraw(self, amount):
        if mock_wallet_transaction(self.wallet_id, -amount):
            logging.info(f"Withdrew {amount} from wallet {self.wallet_id}")
        else:
            logging.error(f"Failed to withdraw {amount} from wallet {self.wallet_id}")

    def get_balance(self):
        return mock_wallet_balance(self.wallet_id)

# Usage:
config = {'wallet_id': 'mock_wallet_1'}
mock_wallet = MockWallet(config)
mock_wallet.deposit(100)
print(mock_wallet.get_balance())
