# Full file path: /moneyverse/managers/database_manager.py

import os
import mysql.connector
from mysql.connector import Error
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import aiosqlite
import asyncio
import logging
from datetime import datetime, timedelta

# Load environment variables for secure credentials
load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

# Encryption Setup
encryption_key = os.getenv("ENCRYPTION_KEY") or Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

# Database configuration with SSL for MySQL
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
    "ssl_ca": os.getenv("MYSQL_SSL_CA"),  # SSL certificate authority path
    "ssl_cert": os.getenv("MYSQL_SSL_CERT"),  # SSL certificate path
    "ssl_key": os.getenv("MYSQL_SSL_KEY")     # SSL key path
}

# --- Helper Functions for Encryption ---

def encrypt_data(data: str) -> str:
    """Encrypts data using Fernet encryption."""
    return cipher_suite.encrypt(data.encode()).decode()

def decrypt_data(data: str) -> str:
    """Decrypts data using Fernet encryption."""
    return cipher_suite.decrypt(data.encode()).decode()

# --- MySQL Connection Class for Synchronous Operations ---

class DatabaseManager:
    """Handles MySQL database connections and operations for synchronous data storage and retrieval."""
    def __init__(self):
        self.connection = self.connect_db()

    def connect_db(self):
        """Establish a secure connection to the MySQL database."""
        try:
            connection = mysql.connector.connect(**DB_CONFIG)
            if connection.is_connected():
                logger.info("Connected to MySQL database.")
            return connection
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def close_connection(self):
        """Close the MySQL connection."""
        if self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed.")

    def log_trade(self, trade_data):
        """Log trade data to the MySQL database."""
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

    def add_to_list(self, list_type, token_address, additional_data=None):
        """Add a token to a specified list (whitelist/blacklist) in the MySQL database."""
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
        """Remove a token from a specified list (whitelist/blacklist) in the MySQL database."""
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

# --- Asynchronous SQLite Operations for Caching and Performance ---

# Cache configuration for frequently accessed data
_cache = {"pumplist": None, "redlist": None, "cache_timestamp": None}
CACHE_DURATION = timedelta(minutes=10)

async def connect_db():
    """Connect to the SQLite database asynchronously."""
    return await aiosqlite.connect('moneyverse.db')

async def refresh_cache():
    """Refreshes the cache with the latest entries from the database."""
    global _cache
    async with connect_db() as db:
        _cache["pumplist"] = await load_pumplist_from_db()
        _cache["redlist"] = await load_redlist_from_db()
        _cache["cache_timestamp"] = datetime.now()

async def get_cached_pumplist():
    """Retrieve the pumplist from cache or refresh it if the cache is outdated."""
    if _cache["pumplist"] is None or datetime.now() - _cache["cache_timestamp"] > CACHE_DURATION:
        await refresh_cache()
    return _cache["pumplist"]

async def get_cached_redlist():
    """Retrieve the redlist from cache or refresh it if the cache is outdated."""
    if _cache["redlist"] is None or datetime.now() - _cache["cache_timestamp"] > CACHE_DURATION:
        await refresh_cache()
    return _cache["redlist"]

# --- Asynchronous Database Operations for List Management ---

async def add_pumplist_entry(entry_id, focus_duration, strategies):
    """Adds an entry to the Pumplist in the database."""
    expiry_timestamp = datetime.now().timestamp() + focus_duration * 60
    async with connect_db() as db:
        await db.execute(
            "INSERT OR IGNORE INTO pumplist (entry_id, strategies, focus_duration, expiry_timestamp, status) "
            "VALUES (?, ?, ?, ?, ?)", (entry_id, ','.join(strategies), focus_duration, expiry_timestamp, 'active')
        )
        await db.commit()
    await refresh_cache()  # Refresh cache after DB update

async def remove_pumplist_entry(entry_id):
    """Removes an entry from the Pumplist in the database."""
    async with connect_db() as db:
        await db.execute("DELETE FROM pumplist WHERE entry_id = ?", (entry_id,))
        await db.commit()
    await refresh_cache()  # Refresh cache after DB update

async def add_redlist_entry(bad_actor):
    """Adds a bad actor to the Redlist in the database."""
    date_added = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    async with connect_db() as db:
        await db.execute("INSERT OR IGNORE INTO redlist (bad_actor, date_added) VALUES (?, ?)", (bad_actor, date_added))
        await db.commit()
    await refresh_cache()  # Refresh cache after DB update

async def log_performance_metric(metric_name, value):
    """Logs a performance metric into the database."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    async with connect_db() as db:
        await db.execute("INSERT INTO performance_metrics (metric_name, value, timestamp) VALUES (?, ?, ?)", 
                         (metric_name, value, timestamp))
        await db.commit()

# --- Load Data from Database ---

async def load_pumplist_from_db():
    """Loads the Pumplist entries from the database."""
    async with connect_db() as db:
        cursor = await db.execute("SELECT entry_id, strategies, focus_duration, expiry_timestamp, status FROM pumplist")
        rows = await cursor.fetchall()
        return [{"entry": row[0], "strategies": row[1].split(','), "duration": row[2], "expiry_timestamp": row[3], "status": row[4]} for row in rows]

async def load_redlist_from_db():
    """Loads the Redlist entries from the database."""
    async with connect_db() as db:
        cursor = await db.execute("SELECT bad_actor, date_added FROM redlist")
        rows = await cursor.fetchall()
        return [{"bad_actor": row[0], "date_added": row[1]} for row in rows]
