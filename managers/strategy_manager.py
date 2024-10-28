import asyncio
import mysql.connector
from mysql.connector import Error
from centralized_logger import CentralizedLogger
from src.list_manager import ListManager
from src.utils.error_handler import handle_errors

logger = CentralizedLogger()
list_manager = ListManager()

class StrategyManager:
    """
    Manages the loading, validation, and storage of trading strategies.
    """

    def __init__(self, db_config):
        self.strategies = []
        self.db_config = db_config
        self.connection = self.connect_db()

    def connect_db(self):
        """Establishes a database connection using provided configurations."""
        try:
            connection = mysql.connector.connect(
                host=self.db_config["host"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                database=self.db_config["database"]
            )
            if connection.is_connected():
                logger.log("info", "Connected to the strategies database.")
            return connection
        except Error as e:
            logger.log("error", f"Error connecting to database: {e}")
            raise

    async def load_strategies(self):
        """
        Load and validate trading strategies asynchronously.
        """
        logger.log("info", "Loading trading strategies...")
        try:
            # Fetch strategies from the database
            strategies = await self.fetch_strategies_from_db()

            # Validate and store only valid strategies
            valid_strategies = [
                strategy for strategy in strategies if self.is_valid_strategy(strategy)
            ]

            self.strategies = valid_strategies
            logger.log("info", f"Loaded {len(valid_strategies)} valid strategies.")
            return valid_strategies

        except Exception as e:
            logger.log("error", f"Error loading strategies: {e}")
            handle_errors(e)

    def is_valid_strategy(self, strategy):
        """
        Validate a strategy based on list management (e.g., blacklist).
        """
        if list_manager.is_blacklisted(strategy["name"]):
            logger.log("warning", f"Strategy {strategy['name']} is blacklisted. Skipping.")
            return False
        return True

    async def fetch_strategies_from_db(self):
        """
        Fetch trading strategies from a database asynchronously.
        """
        query = "SELECT name, interval FROM strategies"
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            strategies = cursor.fetchall()
            cursor.close()
            logger.log("info", f"Fetched {len(strategies)} strategies from the database.")
            return strategies
        except Error as e:
            logger.log("error", f"Error fetching strategies from database: {e}")
            handle_errors(e)
            return []
