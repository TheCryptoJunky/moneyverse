# Full file path: /moneyverse/all_logging/centralized_logger.py

import logging
import os
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables for secure access
load_dotenv()

class CentralizedLogger:
    """
    Centralized logging system that logs either to rotating log files or MySQL database based on configuration.
    """
    def __init__(self, log_file='moneyverse.log'):
        self.logging_method = os.getenv("LOGGING_METHOD", "database")  # Options: 'database' or 'file'
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.logger = logging.getLogger('centralized_logger')
        self.logger.setLevel(self.log_level)
        self.log_file = log_file
        self.connection = None

        self.setup_logger()

        # Setup MySQL connection if logging method is database
        if self.logging_method == "database":
            self.connect_to_db()

    def setup_logger(self):
        """Sets up file-based logging with rotation if required by configuration."""
        if self.logging_method == "file":
            # File logging with rotation
            log_directory = 'logs'
            os.makedirs(log_directory, exist_ok=True)
            log_path = os.path.join(log_directory, self.log_file)

            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=10_000_000, backupCount=5
            )
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            self.logger.info("File-based logging initialized.")
        else:
            self.logger.info("Configured for database-based logging.")

    def connect_to_db(self):
        """Establishes a connection to the MySQL database for logging if configured."""
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv('MYSQL_HOST'),
                user=os.getenv('MYSQL_USER'),
                password=os.getenv('MYSQL_PASSWORD'),
                database=os.getenv('MYSQL_DATABASE'),
                ssl_ca=os.getenv("MYSQL_SSL_CA"),
                ssl_cert=os.getenv("MYSQL_SSL_CERT"),
                ssl_key=os.getenv("MYSQL_SSL_KEY")
            )
            if self.connection.is_connected():
                self.logger.info("Connected to MySQL for logging.")
        except Error as e:
            self.logger.error(f"Error connecting to MySQL: {e}")

    def log(self, level, message):
        """General-purpose logging method that directs logs to the appropriate output based on configuration."""
        if self.logging_method == "file":
            getattr(self.logger, level.lower())(message)
        else:
            self._log_to_db(level.upper(), message)

    def _log_to_db(self, log_type, message):
        """Logs messages to the MySQL database."""
        if not self.connection or not self.connection.is_connected():
            self.connect_to_db()
        
        query = "INSERT INTO bot_logs (log_type, message, timestamp) VALUES (%s, %s, NOW())"
        values = (log_type, message)
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, values)
            self.connection.commit()
            cursor.close()
        except Error as e:
            self.logger.error(f"Error logging to MySQL: {e}")

    # Convenience methods for different log levels
    def log_info(self, message):
        self.log("info", message)

    def log_warning(self, message):
        self.log("warning", message)

    def log_error(self, message):
        self.log("error", message)

    def log_critical(self, message):
        self.log("critical", message)

    def log_event(self, event_type, entry_id, description=""):
        """Log specific events, with an event type and optional description, formatted for readability."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        event_message = f"{event_type} | Entry ID: {entry_id} | {description} at {timestamp}"
        self.log_info(event_message)

    def close(self):
        """Closes MySQL connection safely."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.logger.info("MySQL connection closed.")
