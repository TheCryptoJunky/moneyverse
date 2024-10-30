import mysql.connector
import logging
from typing import Dict

class DatabaseConnection:
    """
    Manages database connections and operations for wallet and NAV data.
    
    Attributes:
    - connection (mysql.connector.connection_cext.CMySQLConnection): Database connection instance.
    """

    def __init__(self, host: str, user: str, password: str, database: str):
        self.connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Database connection established.")

    def save_wallet(self, wallet_data: Dict[str, any]):
        """
        Saves wallet information to the database.

        Args:
        - wallet_data (dict): Data to save, including wallet address and encrypted recovery phrase.
        """
        cursor = self.connection.cursor()
        query = """
        INSERT INTO wallets (address, recovery_phrase, initial_balance) 
        VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE initial_balance = VALUES(initial_balance)
        """
        cursor.execute(query, (
            wallet_data["address"],
            wallet_data["recovery_phrase"],
            wallet_data["initial_balance"]
        ))
        self.connection.commit()
        cursor.close()
        self.logger.info(f"Wallet {wallet_data['address']} saved to database.")

    def update_wallet_balance(self, address: str, balance: float):
        """
        Updates wallet balance in the database.

        Args:
        - address (str): Wallet address.
        - balance (float): New balance to update.
        """
        cursor = self.connection.cursor()
        query = "UPDATE wallets SET balance = %s WHERE address = %s"
        cursor.execute(query, (balance, address))
        self.connection.commit()
        cursor.close()
        self.logger.info(f"Updated balance for wallet {address} to {balance}.")

    def get_wallet_data(self, address: str) -> Dict[str, any]:
        """
        Retrieves wallet data from the database.

        Args:
        - address (str): Wallet address to retrieve data for.

        Returns:
        - dict: Wallet data including encrypted recovery phrase and balance.
        """
        cursor = self.connection.cursor(dictionary=True)
        query = "SELECT * FROM wallets WHERE address = %s"
        cursor.execute(query, (address,))
        wallet_data = cursor.fetchone()
        cursor.close()
        self.logger.info(f"Retrieved data for wallet {address}.")
        return wallet_data

    def update_swarm_nav(self, total_nav: float):
        """
        Updates the total Net Asset Value (NAV) of the wallet swarm in the database.

        Args:
        - total_nav (float): The updated NAV value.
        """
        cursor = self.connection.cursor()
        query = "UPDATE swarm_info SET total_nav = %s WHERE id = 1"
        cursor.execute(query, (total_nav,))
        self.connection.commit()
        cursor.close()
        self.logger.info(f"Swarm NAV updated to {total_nav}.")
    
    def close(self):
        """
        Closes the database connection.
        """
        self.connection.close()
        self.logger.info("Database connection closed.")
