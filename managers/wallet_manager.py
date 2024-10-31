# moneyverse/managers/wallet_manager.py

import logging
from typing import Dict, Optional

class WalletManager:
    """
    Manages wallets and assets, updates balances, and handles multi-chain operations.

    Attributes:
    - wallets (dict): Dictionary to store wallet addresses and associated assets.
    - logger (Logger): Logs actions related to wallet management.
    """

    def __init__(self):
        self.wallets = {}  # Stores wallets with asset balances per chain
        self.logger = logging.getLogger(__name__)
        self.logger.info("WalletManager initialized with empty wallets.")

    def add_wallet(self, address: str, initial_balances: Optional[Dict[str, float]] = None):
        """
        Adds a new wallet with optional initial balances.

        Args:
        - address (str): Wallet address.
        - initial_balances (dict): Initial asset balances, keyed by asset type.
        """
        self.wallets[address] = initial_balances or {}
        self.logger.info(f"Added new wallet {address} with balances: {self.wallets[address]}")

    def remove_wallet(self, address: str):
        """
        Removes a wallet from management.

        Args:
        - address (str): Wallet address to remove.
        """
        if address in self.wallets:
            del self.wallets[address]
            self.logger.info(f"Removed wallet {address}")
        else:
            self.logger.warning(f"Attempted to remove non-existent wallet {address}")

    def update_balance(self, address: str, asset: str, amount: float):
        """
        Updates the balance of a specified asset within a wallet.

        Args:
        - address (str): Wallet address.
        - asset (str): Asset type (e.g., "ETH", "BTC").
        - amount (float): Amount to add (positive) or subtract (negative).
        """
        if address not in self.wallets:
            self.logger.error(f"Wallet {address} does not exist.")
            return

        if asset in self.wallets[address]:
            self.wallets[address][asset] += amount
        else:
            self.wallets[address][asset] = amount
        self.logger.info(f"Updated balance for {asset} in wallet {address}: {self.wallets[address][asset]}")

    def get_balance(self, address: str, asset: str) -> Optional[float]:
        """
        Retrieves the balance of a specified asset in a wallet.

        Args:
        - address (str): Wallet address.
        - asset (str): Asset type to retrieve balance for.

        Returns:
        - float: Balance of the specified asset, or None if wallet/asset not found.
        """
        if address in self.wallets and asset in self.wallets[address]:
            balance = self.wallets[address][asset]
            self.logger.debug(f"Balance for {asset} in wallet {address}: {balance}")
            return balance
        self.logger.warning(f"Balance for {asset} in wallet {address} not found.")
        return None

    def list_wallets(self) -> Dict[str, Dict[str, float]]:
        """
        Lists all wallets and their balances.

        Returns:
        - dict: Dictionary of wallets with their balances.
        """
        self.logger.info("Listing all wallets and balances.")
        return self.wallets

    def transfer_asset(self, from_address: str, to_address: str, asset: str, amount: float):
        """
        Transfers an asset from one wallet to another.

        Args:
        - from_address (str): Source wallet address.
        - to_address (str): Destination wallet address.
        - asset (str): Asset type to transfer.
        - amount (float): Amount to transfer.
        """
        if from_address not in self.wallets or to_address not in self.wallets:
            self.logger.error(f"One or both wallets not found for transfer: {from_address} to {to_address}")
            return

        if self.get_balance(from_address, asset) >= amount:
            self.update_balance(from_address, asset, -amount)
            self.update_balance(to_address, asset, amount)
            self.logger.info(f"Transferred {amount} {asset} from {from_address} to {to_address}.")
        else:
            self.logger.warning(f"Insufficient funds in {from_address} for transfer of {amount} {asset}.")

    def initialize_wallets(self, wallet_data: Dict[str, Dict[str, float]]):
        """
        Bulk initializes wallets with data, helpful for setting up multiple wallets at once.

        Args:
        - wallet_data (dict): Dictionary of wallet addresses and initial balances.
        """
        for address, balances in wallet_data.items():
            self.add_wallet(address, balances)
        self.logger.info("Bulk wallet initialization complete.")
