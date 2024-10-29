# File: /src/database/mysql_setup.py

import subprocess
import mysql.connector
from mysql.connector import Error # type: ignore
import all_logging.centralized_logger
import logging
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
dotenv_path = find_dotenv()
if not dotenv_path:
    logging.warning(".env file not found. Make sure environment variables are set.")
else:
    load_dotenv(dotenv_path)

# Setup logging
logging.basicConfig(level=logging.INFO)

def install_mysql():
    """
    Installs MySQL server if it is not already installed.
    """
    try:
        logging.info("Checking if MySQL is installed...")
        result = subprocess.run(["mysql", "--version"], capture_output=True, text=True)
        if "mysql" in result.stdout.lower():
            logging.info("MySQL is already installed.")
        else:
            logging.info("Installing MySQL server...")
            subprocess.run(["sudo", "apt-get", "install", "mysql-server", "-y"], check=True)
            logging.info("MySQL installation completed.")
    except subprocess.CalledProcessError as err:
        logging.error(f"Error installing MySQL: {err}")
        exit(1)

def connect_to_db():
    """
    Connect to MySQL using credentials from the environment variables.
    """
    try:
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE')
        )
        logging.info("Successfully connected to MySQL.")
        return conn, conn.cursor()
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        raise

def create_user_and_database(cursor):
    """
    Creates the MySQL user and database if they don't exist.
    """
    try:
        user = os.getenv('MYSQL_USER')
        password = os.getenv('MYSQL_PASSWORD')
        database = os.getenv('MYSQL_DATABASE')

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database};")
        cursor.execute(f"CREATE USER IF NOT EXISTS '{user}'@'localhost' IDENTIFIED BY '{password}';")
        cursor.execute(f"GRANT ALL PRIVILEGES ON {database}.* TO '{user}'@'localhost';")
        cursor.execute("FLUSH PRIVILEGES;")
        logging.info(f"Database '{database}' and user '{user}' created successfully.")
    except Error as e:
        logging.error(f"Error creating database or user: {e}")
        raise

def create_tables(cursor):
    """
    Creates the necessary tables for the trading bots, including logs, trades, and lists (whitelist, blacklist, etc.).
    """
    try:
        # Create 'trades' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(10),
                trade_action VARCHAR(10),
                trade_size DECIMAL(18, 8),
                price DECIMAL(18, 8),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logging.info("Table 'trades' created successfully.")

        # Enhanced 'bot_logs' table for centralized logging
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bot_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                bot_name VARCHAR(255),
                identifier VARCHAR(255),
                log_type VARCHAR(50),
                severity VARCHAR(10), -- Log level like INFO, WARNING, ERROR
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logging.info("Table 'bot_logs' created successfully.")

        # Create 'whitelist' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS whitelist (
                token_address VARCHAR(255) PRIMARY KEY,
                added_by VARCHAR(50),
                notes TEXT,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logging.info("Table 'whitelist' created successfully.")

        # Create 'blacklist' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blacklist (
                token_address VARCHAR(255) PRIMARY KEY,
                reason TEXT,
                flagged_by VARCHAR(50),
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logging.info("Table 'blacklist' created successfully.")

        # Create 'redlist' table (for aggressive targets)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS redlist (
                token_address VARCHAR(255) PRIMARY KEY,
                reason TEXT,
                flagged_by VARCHAR(50),
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logging.info("Table 'redlist' created successfully.")

        # Create 'pumplist' table (for projects under temporary AI-driven assistance)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pumplist (
                id INT AUTO_INCREMENT PRIMARY KEY,
                token_address VARCHAR(255),
                focus_duration INT,
                added_by VARCHAR(50),
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logging.info("Table 'pumplist' created successfully.")

        # Create 'logs' table
        cursor.execute("""CREATE TABLE logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                log_type VARCHAR(50),
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logging.info("table 'logs' created successfully.")

        # Create transactions table
        cursor.execute("""CREATE TABLE IF NOT EXISTS transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                wallet_id VARCHAR(255),
                trade_type VARCHAR(255),
                amount DECIMAL(20, 8),
                status VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logging.info("table 'transactions' created successfully.")

        # Create agents table
        cursor.execute("""CREATE TABLE IF NOT EXISTS agents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                strategy_name VARCHAR(255),
                status VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logging.info("table 'agents' created successfully.")
          
    except Error as e:
        logging.error(f"Error creating tables: {e}")
        raise

def setup_database():
    """
    Full setup process: Install MySQL, create database, user, and required tables.
    """
    install_mysql()
    conn, cursor = connect_to_db()
    create_user_and_database(cursor)
    create_tables(cursor)
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    setup_database()
