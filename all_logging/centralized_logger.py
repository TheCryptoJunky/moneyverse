# File: /src/logging/centralized_logger.py

import all_logging
from logging.handlers import RotatingFileHandler
import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CentralizedLogger:
    """
    Centralized logging system that logs either to rotating log files or MySQL database based on configuration.
    """
    def __init__(self):
        self.logging_method = os.getenv("LOGGING_METHOD", "database")  # 'database' or 'file'
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.logger = logging.getLogger('centralized_logger')
        self.logger.setLevel(self.log_level)
        self.setup_logger()

        # MySQL configuration for logging
        if self.logging_method == "database":
            try:
                self.connection = mysql.connector.connect(
                    host=os.getenv('MYSQL_HOST'),
                    user=os.getenv('MYSQL_USER'),
                    password=os.getenv('MYSQL_PASSWORD'),
                    database=os.getenv('MYSQL_DATABASE')
                )
                self.cursor = self.connection.cursor()
                logging.info("Connected to MySQL for logging.")
            except Error as e:
                self.logger.error(f"Error connecting to MySQL: {e}")

    def setup_logger(self):
        """Sets up logging to file or database."""
        if self.logging_method == "file":
            # Setup file logging (fallback option)
            log_directory = 'logs'
            os.makedirs(log_directory, exist_ok=True)
            log_file = os.path.join(log_directory, 'trade_log.txt')

            file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)

            self.logger.addHandler(file_handler)
            logging.info("File logging initialized.")
        else:
            logging.info("MySQL logging initialized.")

    def log_trade(self, symbol, buy_order, sell_order, metadata):
        """Logs trade details based on the configured logging method."""
        message = f"Trade executed: {symbol}, Buy Order: {buy_order}, Sell Order: {sell_order}, Metadata: {metadata}"
        if self.logging_method == "file":
            self.logger.info(message)
        else:
            self._log_to_db(symbol, "TRADE", message)

    def log_failure(self, bot_id, message, details):
        """Logs failure based on the configured logging method."""
        error_message = f"Bot {bot_id} failed: {message} - Details: {details}"
        if self.logging_method == "file":
            self.logger.error(error_message)
        else:
            self._log_to_db(bot_id, "FAILURE", error_message)

    def _log_to_db(self, identifier, log_type, message):
        """Logs messages to the MySQL database."""
        sql = "INSERT INTO bot_logs (identifier, log_type, message) VALUES (%s, %s, %s)"
        values = (identifier, log_type, message)
        try:
            self.cursor.execute(sql, values)
            self.connection.commit()
        except Error as e:
            self.logger.error(f"Error logging to MySQL: {e}")

    def close(self):
        """Closes MySQL connection safely."""
        if self.connection.is_connected():
            self.cursor.close()
            self.connection.close()

import mysql.connector
from src.database.database_manager import DatabaseManager
from src.utils.error_handler import handle_errors

class CentralizedLogger:
    def __init__(self):
        self.db_manager = DatabaseManager()  # MySQL connection manager

    def log(self, log_type, message):
        """
        Logs the message to the MySQL database.
        log_type: 'info', 'warning', 'error', 'critical'
        """
        print(f"[{log_type.upper()}] {message}")  # Console output (optional)

        # Insert the log into the MySQL database
        try:
            connection = self.db_manager.get_connection()
            cursor = connection.cursor()

            # Log message to MySQL (table: logs)
            query = """
                INSERT INTO logs (log_type, message, timestamp)
                VALUES (%s, %s, NOW())
            """
            data = (log_type, message)

            cursor.execute(query, data)
            connection.commit()
            cursor.close()

        except mysql.connector.Error as err:
            print(f"[ERROR] Failed to log to MySQL: {err}")
            handle_errors(err)

    def log_error(self, message):
        """
        Convenience method for logging errors.
        """
        self.log("error", message)

    def log_info(self, message):
        """
        Convenience method for logging informational messages.
        """
        self.log("info", message)

    def log_warning(self, message):
        """
        Convenience method for logging warnings.
        """
        self.log("warning", message)

    def log_critical(self, message):
        """
        Convenience method for logging critical errors.
        """
        self.log("critical", message)
