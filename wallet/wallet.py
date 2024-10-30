import logging
from decimal import Decimal
from typing import Dict, Any
from cryptography.fernet import Fernet  # For encrypting sensitive data
from database.db_connection import DatabaseConnection

class Wallet:
    def __init__(self, address: str, recovery_phrase: str, db: DatabaseConnection, key: bytes):
        """Initialize wallet with address, encrypted recovery phrase, and an empty balance"""
        self.address = address
        self.db = db
        self.encryption = Fernet(key)
        self.recovery_phrase = self.encrypt_data(recovery_phrase)
        self.assets = {}  # {asset_symbol: {"balance": Decimal, "usd_value": Decimal}}
        self.nfts = []  # List of NFTs with value and metadata
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Wallet initialized for address: {self.address}")
        self.save_to_db()

    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive information."""
        return self.encryption.encrypt(data.encode())

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive information."""
        return self.encryption.decrypt(encrypted_data).decode()

    def add_asset(self, asset_symbol: str, balance: Decimal, usd_value_per_unit: Decimal):
        """Add or update asset in wallet."""
        self.assets[asset_symbol] = {
            "balance": balance,
            "usd_value": balance * usd_value_per_unit
        }
        self.logger.info(f"Updated {asset_symbol} balance to {balance}, USD value {self.assets[asset_symbol]['usd_value']}")
        self.update_db()

    def add_nft(self, nft_data: Dict[str, Any], value_in_native_token: Decimal, usd_value: Decimal):
        """Add an NFT with metadata."""
        self.nfts.append({
            "data": nft_data,
            "value_native": value_in_native_token,
            "usd_value": usd_value
        })
        self.logger.info(f"Added NFT with value {usd_value} USD.")
        self.update_db()

    def calculate_total_nav(self) -> Dict[str, Decimal]:
        """Calculate and return total NAV (Net Asset Value) of the wallet in both tokens and USD."""
        token_value = sum(asset["balance"] for asset in self.assets.values())
        usd_value = sum(asset["usd_value"] for asset in self.assets.values())
        nft_value = sum(nft["usd_value"] for nft in self.nfts)
        total_nav = usd_value + nft_value
        return {"token_value": token_value, "usd_value": usd_value, "nft_value": nft_value, "total_nav": total_nav}

    def save_to_db(self):
        """Save wallet data to the database securely."""
        encrypted_phrase = self.recovery_phrase
        wallet_data = {
            "address": self.address,
            "recovery_phrase": encrypted_phrase,
            "assets": self.assets,
            "nfts": self.nfts
        }
        self.db.save_wallet(wallet_data)

    def update_db(self):
        """Update wallet data in the database."""
        nav_data = self.calculate_total_nav()
        self.db.update_wallet(self.address, self.assets, self.nfts, nav_data)

    def get_balance(self, asset_symbol: str) -> Decimal:
        """Retrieve balance of a specific asset."""
        return self.assets.get(asset_symbol, {}).get("balance", Decimal(0))
