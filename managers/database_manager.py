# Full file path: /moneyverse/managers/database_manager.py

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os
import logging

# Load environment variables for database credentials
load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Handles database connections, trade logging, and list management (whitelist, blacklist, etc.) for trading bots.
    """

    def __init__(self):
        """Initializes the DatabaseManager with MySQL configuration and connects to the database."""
        self.connection = self.connect_db()

    def connect_db(self):
        """Establish a connection to the MySQL database using environment variables."""
        try:
            connection = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST"),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DATABASE")
            )
            if connection.is_connected():
                logger.info("Connected to MySQL database.")
            return connection
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def close_connection(self):
        """Close the MySQL connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed.")

    def log_trade(self, trade_data):
        """
        Log trade data to the MySQL database.
        """
        query = """
        INSERT INTO trades (symbol, trade_action, trade_size, price, timestamp)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, (trade_data['symbol'], trade_data['action'], trade_data['size'],
                                   trade_data['price'], trade_data['timestamp']))
            self.connection.commit()
            logger.info(f"Trade logged: {trade_data}")
        except Error as e:
            logger.error(f"Error logging trade: {e}")
        finally:
            cursor.close()

    def fetch_list(self, list_type):
        """
        Fetch the specified list (e.g., whitelist, blacklist) from the database.
        """
        query = f"SELECT * FROM {list_type}"
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            result = cursor.fetchall()
            logger.info(f"Fetched {len(result)} entries from {list_type}.")
            return result
        except Error as e:
            logger.error(f"Error fetching {list_type}: {e}")
            return []
        finally:
            cursor.close()

    def add_to_list(self, list_type, token_address, additional_data=None):
        """
        Add a token to a specified list in the database.
        """
        query = f"INSERT INTO {list_type} (token_address, additional_data) VALUES (%s, %s)"
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, (token_address, additional_data))
            self.connection.commit()
            logger.info(f"Added {token_address} to {list_type}.")
        except Error as e:
            logger.error(f"Error adding to {list_type}: {e}")
        finally:
            cursor.close()

    def remove_from_list(self, list_type, token_address):
        """
        Remove a token from a specified list in the database.
        """
        query = f"DELETE FROM {list_type} WHERE token_address = %s"
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, (token_address,))
            self.connection.commit()
            logger.info(f"Removed {token_address} from {list_type}.")
        except Error as e:
            logger.error(f"Error removing from {list_type}: {e}")
        finally:
            cursor.close()

# Ensure connection closure on exit
if __name__ == "__main__":
    db_manager = DatabaseManager()
    try:
        # Example of logging a trade
        trade = {
            'symbol': 'BTC/USD',
            'action': 'buy',
            'size': 1.5,
            'price': 60000,
            'timestamp': '2023-10-30 15:30:00'
        }
        db_manager.log_trade(trade)
    finally:
        db_manager.close_connection()
