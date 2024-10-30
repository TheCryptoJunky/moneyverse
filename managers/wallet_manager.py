import logging
from typing import List, Dict
from cryptography.fernet import Fernet  # For encryption of sensitive data
from ..wallet.wallet import Wallet
from ..database.db_connection import DatabaseConnection

class WalletManager:
    """
    Manages wallets, including their initialization, asset balancing, and secure storage.

    Attributes:
    - db (DatabaseConnection): Database connection instance for storage.
    - encryption_key (bytes): Key for encrypting sensitive wallet information.
    - wallets (Dict[str, Wallet]): Active wallets managed by this manager.
    """

    def __init__(self, db: DatabaseConnection, encryption_key: bytes):
        self.db = db
        self.encryption = Fernet(encryption_key)
        self.wallets = {}
        self.logger = logging.getLogger(__name__)

    def register_wallet(self, wallet: Wallet):
        """
        Registers a new wallet, storing it securely in the database.
        
        Args:
        - wallet (Wallet): Wallet instance to register.
        """
        encrypted_phrase = self.encrypt(wallet.recovery_phrase)
        self.wallets[wallet.address] = wallet
        self.db.save_wallet({
            "address": wallet.address,
            "recovery_phrase": encrypted_phrase,
            "initial_balance": wallet.get_balance(),
        })
        self.logger.info(f"Registered wallet {wallet.address} with initial balance {wallet.get_balance()}.")

    def encrypt(self, data: str) -> bytes:
        """Encrypts sensitive data."""
        return self.encryption.encrypt(data.encode())

    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypts sensitive data."""
        return self.encryption.decrypt(encrypted_data).decode()

    def redistribute_assets(self, target_wallet: Wallet, amount: float):
        """
        Rebalances assets by transferring the specified amount to the target wallet.
        
        Args:
        - target_wallet (Wallet): Wallet to receive the assets.
        - amount (float): Amount to adjust.
        """
        # Placeholder for actual transfer logic
        current_balance = target_wallet.get_balance()
        target_wallet.update_balance(current_balance + amount)
        self.db.update_wallet_balance(target_wallet.address, target_wallet.get_balance())
        self.logger.info(f"Redistributed {amount} to wallet {target_wallet.address}. New balance: {target_wallet.get_balance()}")

    def get_wallets(self) -> List[Wallet]:
        """
        Retrieves all active wallets managed by this manager.
        
        Returns:
        - list: All wallet instances.
        """
        return list(self.wallets.values())
    
    def secure_retrieve_wallet(self, address: str) -> Wallet:
        """
        Retrieves a wallet by address, decrypting sensitive information.
        
        Args:
        - address (str): Wallet address to retrieve.
        
        Returns:
        - Wallet: The decrypted wallet instance.
        """
        encrypted_data = self.db.get_wallet_data(address)
        recovery_phrase = self.decrypt(encrypted_data["recovery_phrase"])
        wallet = Wallet(address=address, recovery_phrase=recovery_phrase)
        self.logger.info(f"Retrieved wallet {address} securely.")
        return wallet
