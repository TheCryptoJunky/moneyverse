import logging
from typing import Dict

class Wallet:
    """
    Manages wallet operations, including balance tracking, asset conversion, and strategy assignment.

    Attributes:
    - address (str): Wallet's blockchain address.
    - assets (dict): Maps asset type to balance.
    - strategy (str): Current trading strategy assigned to the wallet.
    """

    def __init__(self, address: str, initial_assets: Dict[str, float] = None):
        self.address = address
        self.assets = initial_assets or {}
        self.strategy = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized wallet {address}.")

    def update_balance(self, asset_type: str, amount: float):
        """
        Updates the balance for a specified asset type.

        Args:
        - asset_type (str): Type of asset (e.g., 'BTC', 'ETH').
        - amount (float): Amount to add or remove.
        """
        self.assets[asset_type] = self.assets.get(asset_type, 0.0) + amount
        self.logger.info(f"Updated balance for {asset_type} in wallet {self.address}: {self.assets[asset_type]}.")

    def get_balance(self, asset_type: str) -> float:
        """
        Retrieves the balance for a specified asset type.

        Args:
        - asset_type (str): Type of asset.

        Returns:
        - float: Current balance of the asset type.
        """
        return self.assets.get(asset_type, 0.0)

    def calculate_total_value(self, asset_prices: Dict[str, float]) -> float:
        """
        Calculates the wallet's total value in USD based on asset prices.

        Args:
        - asset_prices (dict): Maps asset types to their USD values.

        Returns:
        - float: Total value of wallet in USD.
        """
        total_value = sum(self.get_balance(asset) * asset_prices.get(asset, 0.0) for asset in self.assets)
        self.logger.info(f"Total value of wallet {self.address} calculated: {total_value} USD.")
        return total_value

    def assign_strategy(self, strategy: str):
        """
        Assigns a trading strategy to the wallet.

        Args:
        - strategy (str): Name of the strategy to assign.
        """
        self.strategy = strategy
        self.logger.info(f"Strategy {strategy} assigned to wallet {self.address}.")

    def transfer_asset(self, recipient_wallet: 'Wallet', asset_type: str, amount: float) -> bool:
        """
        Transfers an asset to another wallet.

        Args:
        - recipient_wallet (Wallet): Wallet receiving the asset.
        - asset_type (str): Type of asset to transfer.
        - amount (float): Amount to transfer.

        Returns:
        - bool: True if transfer is successful, False otherwise.
        """
        if self.get_balance(asset_type) >= amount:
            self.update_balance(asset_type, -amount)
            recipient_wallet.update_balance(asset_type, amount)
            self.logger.info(f"Transferred {amount} {asset_type} from {self.address} to {recipient_wallet.address}.")
            return True
        else:
            self.logger.warning(f"Transfer failed: insufficient {asset_type} balance in wallet {self.address}.")
            return False

    def swap_asset(self, from_asset: str, to_asset: str, amount: float, conversion_rate: float):
        """
        Swaps one asset for another within the wallet.

        Args:
        - from_asset (str): Asset type to convert from.
        - to_asset (str): Asset type to convert to.
        - amount (float): Amount of the from_asset to swap.
        - conversion_rate (float): Rate to apply for the swap.
        """
        if self.get_balance(from_asset) >= amount:
            self.update_balance(from_asset, -amount)
            converted_amount = amount * conversion_rate
            self.update_balance(to_asset, converted_amount)
            self.logger.info(f"Swapped {amount} {from_asset} to {converted_amount} {to_asset} in wallet {self.address}.")
        else:
            self.logger.warning(f"Swap failed: insufficient {from_asset} balance in wallet {self.address}.")
