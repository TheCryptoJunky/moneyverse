from prometheus_client import start_http_server, Summary, Counter, Gauge
import mysql.connector
from mysql.connector import Error
import time
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Create Prometheus metrics
REQUEST_TIME = Summary('bot_processing_seconds', 'Time spent processing bot requests')
TRANSACTION_COUNT = Counter('transaction_total', 'Total number of transactions recorded')
BOT_STATUS = Gauge('bot_status', 'Current status of each bot (0: Stopped, 1: Running)')

# MySQL connection setup
def connect_to_database():
    """Establishes a connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="your_password",
            database="crypto_trading_bot"
        )
        return connection
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        return None

def process_bot(bot_id):
    """
    Simulates bot processing and records the transaction in MySQL.
    Uses Prometheus metrics to track performance.
    """
    with REQUEST_TIME.time():  # Track processing time
        BOT_STATUS.set(1)  # Set bot status as 'running'

        # Record transaction in MySQL
        connection = connect_to_database()
        if connection:
            cursor = connection.cursor()
            sql = "INSERT INTO transactions (bot_id, tx_hash, status) VALUES (%s, %s, %s)"
            values = (bot_id, "tx_hash_placeholder", "SUCCESS")

            try:
                cursor.execute(sql, values)
                connection.commit()
                TRANSACTION_COUNT.inc()  # Increment transaction count
                logging.info(f"Transaction recorded for bot {bot_id}.")
            except Error as e:
                logging.error(f"Error inserting transaction into MySQL: {e}")
            finally:
                cursor.close()
                connection.close()
        else:
            logging.error("Failed to process transaction. Database connection unavailable.")

        # Simulate bot stopping
        time.sleep(1)
        BOT_STATUS.set(0)  # Set bot status as 'stopped'

if __name__ == '__main__':
    # Start the Prometheus server on port 8000
    start_http_server(8000)
    logging.info("Prometheus server started on port 8000.")

    # Simulate continuous bot processing
    while True:
        process_bot(bot_id="bot_1")
        time.sleep(5)  # Wait 5 seconds between bot runs
