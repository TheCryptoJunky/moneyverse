# File: /src/database/database_manager.py

import mysql.connector
import logging
from mysql.connector import Error
from dotenv import load_dotenv
import os

# Load environment variables for DB credentials
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Centralized DatabaseManager that handles database connections, list management (whitelist, blacklist, etc.),
    and trade logging for the trading bots.
    """

    def __init__(self):
        """Initializes the DatabaseManager with the MySQL configuration."""
        self.connection = self.connect_db()

    def connect_db(self):
        """Connect to the MySQL database using environment variables."""
        try:
            return mysql.connector.connect(
                host=os.getenv("MYSQL_HOST"),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DATABASE")
            )
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def close_connection(self):
        """Closes the MySQL connection."""
        if self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed.")

    def log_trade(self, trade_data):
        """
        Logs trade data to the MySQL database.
        """
        query = """
        INSERT INTO trades (symbol, trade_action, trade_size, price, timestamp)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, (trade_data['symbol'], trade_data['action'], trade_data['size'], trade_data['price'], trade_data['timestamp']))
            self.connection.commit()
            logger.info(f"Trade logged: {trade_data}")
        except Error as e:
            logger.error(f"Error logging trade: {e}")
        finally:
            cursor.close()

    def fetch_list(self, list_type):
        """
        Fetches the list from the database (whitelist, blacklist, greenlist, etc.).
        """
        query = f"SELECT * FROM {list_type}"
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result
        except Error as e:
            logger.error(f"Error fetching {list_type}: {e}")
            return []

    def add_to_list(self, list_type, token_address, additional_data=None):
        """
        Adds a token to a specified list in the database (whitelist, blacklist, etc.).
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
        Removes a token from a specified list in the database (whitelist, blacklist, etc.).
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
