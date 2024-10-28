# Full file path: /moneyverse/managers/strategy_manager.py

import asyncio
import mysql.connector
import json
from centralized_logger import CentralizedLogger
from src.list_manager import ListManager
from src.utils.error_handler import handle_errors
from utils.ai_selection.source_selector import SourceSelector

logger = CentralizedLogger()
list_manager = ListManager()

class StrategyManager:
    """
    Manages the loading, validation, and storage of trading strategies, using SourceSelector to optimize
    data retrieval from multiple sources based on real-time metrics.
    """

    def __init__(self, db_config, api_configs, config_file_path):
        """
        Initializes StrategyManager with database, API configurations, and a config file path.
        
        Parameters:
            db_config (dict): Database connection settings.
            api_configs (list): List of API configurations, each containing details like name, url, and rate limits.
            config_file_path (str): Path to a local configuration file with predefined strategies.
        """
        self.strategies = []
        self.db_config = db_config
        self.config_file_path = config_file_path
        self.connection = self.connect_db() if db_config else None
        self.source_selector = SourceSelector(api_configs)  # Initialize SourceSelector with API configurations
        self.enabled_sources = {"database": db_config is not None, "config_file": True}

    def connect_db(self):
        """Establishes a database connection using the provided configuration."""
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
        except mysql.connector.Error as e:
            logger.log("error", f"Error connecting to database: {e}")
            return None

    async def load_strategies(self):
        """
        Load and validate trading strategies asynchronously, selecting the best source for APIs dynamically.
        """
        logger.log("info", "Loading trading strategies from enabled sources.")
        loaded_strategies = []

        # Fetch from database if enabled and connected
        if self.enabled_sources["database"] and self.connection:
            loaded_strategies += await self.fetch_strategies_from_db()

        # Fetch from the best API source using SourceSelector
        api_strategies = self.source_selector.get_next_best_source()
        if api_strategies:
            loaded_strategies += api_strategies

        # Fetch from config file if enabled
        if self.enabled_sources["config_file"]:
            loaded_strategies += self.fetch_strategies_from_config()

        # Validate and store only valid strategies
        valid_strategies = [s for s in loaded_strategies if self.is_valid_strategy(s)]
        self.strategies = valid_strategies
        logger.log("info", f"Loaded {len(valid_strategies)} valid strategies.")
        return valid_strategies

    def is_valid_strategy(self, strategy):
        """Check if a strategy is valid and not blacklisted."""
        if list_manager.is_blacklisted(strategy["name"]):
            logger.log("warning", f"Strategy {strategy['name']} is blacklisted. Skipping.")
            return False
        return True

    async def fetch_strategies_from_db(self):
        """Fetch strategies from the database asynchronously."""
        query = "SELECT name, interval FROM strategies"
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            strategies = cursor.fetchall()
            cursor.close()
            logger.log("info", f"Fetched {len(strategies)} strategies from the database.")
            return strategies
        except mysql.connector.Error as e:
            logger.log("error", f"Error fetching strategies from database: {e}")
            handle_errors(e)
            return []

    def fetch_strategies_from_config(self):
        """Load strategies from a local configuration file."""
        try:
            with open(self.config_file_path, 'r') as file:
                strategies = json.load(file)
                logger.log("info", f"Fetched {len(strategies)} strategies from config file.")
                return strategies
        except FileNotFoundError as e:
            logger.log("error", f"Config file not found: {e}")
            handle_errors(e)
            return []

# --- Example Usage of StrategyManager with SourceSelector ---

# Database configuration settings
db_config = {
    "host": "localhost",
    "user": "username",
    "password": "password",
    "database": "strategies_db"
}

# List of API configurations for SourceSelector
api_configs = [
    {"name": "API1", "url": "https://example.com/api1/strategies", "headers": {}, "cost_per_request": 0.05, "rate_limit": 1000},
    {"name": "API2", "url": "https://example.com/api2/strategies", "headers": {}, "cost_per_request": 0.03, "rate_limit": 800}
]

# Path to the configuration file
config_file_path = "strategies_config.json"

# Initialize StrategyManager with all source configurations
strategy_manager = StrategyManager(db_config, api_configs, config_file_path)

# Run load_strategies to dynamically load strategies from the optimal source
if __name__ == "__main__":
    asyncio.run(strategy_manager.load_strategies())
