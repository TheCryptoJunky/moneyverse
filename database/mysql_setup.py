# Full file path: /moneyverse/database/mysql_setup.py

import os
import subprocess
import logging
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv, find_dotenv

# Load environment variables
dotenv_path = find_dotenv()
if not dotenv_path:
    logging.warning(".env file not found. Ensure environment variables are set.")
else:
    load_dotenv(dotenv_path)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Database connection configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}

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
    Connect to MySQL using environment credentials.
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        logging.info("Successfully connected to MySQL.")
        return conn
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        raise

def create_user_and_database(cursor):
    """
    Creates MySQL user and database if they do not exist.
    """
    try:
        user = DB_CONFIG["user"]
        password = DB_CONFIG["password"]
        database = DB_CONFIG["database"]

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database};")
        cursor.execute(f"CREATE USER IF NOT EXISTS '{user}'@'localhost' IDENTIFIED BY '{password}';")
        cursor.execute(f"GRANT ALL PRIVILEGES ON {database}.* TO '{user}'@'localhost';")
        cursor.execute("FLUSH PRIVILEGES;")
        logging.info(f"Database '{database}' and user '{user}' set up successfully.")
    except Error as e:
        logging.error(f"Error creating database or user: {e}")
        raise

def create_tables(cursor):
    """
    Creates necessary tables in the database if they do not exist.
    """
    tables = {
        "trades": """
            CREATE TABLE IF NOT EXISTS trades (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(10),
                trade_action VARCHAR(10),
                trade_size DECIMAL(18, 8),
                price DECIMAL(18, 8),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        "bot_logs": """
            CREATE TABLE IF NOT EXISTS bot_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                bot_name VARCHAR(255),
                identifier VARCHAR(255),
                log_type VARCHAR(50),
                severity VARCHAR(10),
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        "whitelist": """
            CREATE TABLE IF NOT EXISTS whitelist (
                token_address VARCHAR(255) PRIMARY KEY,
                added_by VARCHAR(50),
                notes TEXT,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        "blacklist": """
            CREATE TABLE IF NOT EXISTS blacklist (
                token_address VARCHAR(255) PRIMARY KEY,
                reason TEXT,
                flagged_by VARCHAR(50),
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        "redlist": """
            CREATE TABLE IF NOT EXISTS redlist (
                token_address VARCHAR(255) PRIMARY KEY,
                reason TEXT,
                flagged_by VARCHAR(50),
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        "pumplist": """
            CREATE TABLE IF NOT EXISTS pumplist (
                id INT AUTO_INCREMENT PRIMARY KEY,
                token_address VARCHAR(255),
                focus_duration INT,
                added_by VARCHAR(50),
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        "performance_metrics": """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                metric_name VARCHAR(255),
                value FLOAT,
                timestamp DATETIME
            );
        """,
        "transactions": """
            CREATE TABLE IF NOT EXISTS transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                wallet_id VARCHAR(255),
                trade_type VARCHAR(255),
                amount DECIMAL(20, 8),
                status VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        "agents": """
            CREATE TABLE IF NOT EXISTS agents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                strategy_name VARCHAR(255),
                status VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
    }
    for name, ddl in tables.items():
        cursor.execute(ddl)
        logging.info(f"Table '{name}' created successfully.")

def create_indexes(cursor):
    """
    Creates indexes for frequently queried fields to improve database performance.
    """
    indexes = {
        "pumplist_entry_id": "CREATE INDEX IF NOT EXISTS idx_entry_id ON pumplist(token_address)",
        "redlist_bad_actor": "CREATE INDEX IF NOT EXISTS idx_bad_actor ON redlist(token_address)"
    }
    for idx_name, ddl in indexes.items():
        cursor.execute(ddl)
        logging.info(f"Index '{idx_name}' created successfully.")

def setup_database():
    """
    Full setup process: Installs MySQL, creates database, user, tables, and indexes.
    """
    install_mysql()  # Install MySQL if needed
    conn = connect_to_db()  # Connect to the database
    cursor = conn.cursor()
    
    try:
        create_user_and_database(cursor)  # Create database and user if they don't exist
        create_tables(cursor)  # Create tables
        create_indexes(cursor)  # Create indexes
        conn.commit()
        logging.info("MySQL database setup completed successfully.")
    except Error as e:
        logging.error(f"Error during database setup: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    setup_database()
