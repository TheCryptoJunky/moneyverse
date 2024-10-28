import asyncio
import mysql.connector
import requests
import json
import logging
from flask import Flask, request, jsonify
from ai_source_selector import SourceSelector
from centralized_logger import CentralizedLogger
from src.list_manager import ListManager
from src.utils.error_handler import handle_errors

logger = CentralizedLogger()
list_manager = ListManager()

# Flask app for managing sources dynamically
app = Flask(__name__)

class StrategyManager:
    """
    Manages the loading, validation, and selection of trading strategies from multiple sources.
    Supports real-time switching between database, APIs, and config files, with AI-assisted selection.
    """

    def __init__(self, db_config, api_configs, config_file_path):
        self.strategies = []
        self.db_config = db_config
        self.api_configs = api_configs
        self.config_file_path = config_file_path
        self.enabled_sources = {"database": True, "apis": [], "config_file": True}
        self.connection = None
        self.source_selector = SourceSelector(api_configs)

        if self.enabled_sources["database"]:
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
        except mysql.connector.Error as e:
            logger.log("error", f"Error connecting to database: {e}")
            return None

    async def load_strategies(self):
        """Load strategies dynamically based on enabled sources."""
        logger.log("info", "Loading strategies from enabled sources.")
        loaded_strategies = []

        # Database strategies
        if self.enabled_sources["database"] and self.connection:
            loaded_strategies += await self.fetch_strategies_from_db()

        # API-based strategies
        for api_config in self.enabled_sources["apis"]:
            if api_config["enabled"]:
                loaded_strategies += await self.fetch_strategies_from_api(api_config)

        # Config file strategies
        if self.enabled_sources["config_file"]:
            loaded_strategies += self.fetch_strategies_from_config()

        # Filter valid strategies and store
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
        """Fetch strategies from the database."""
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

    async def fetch_strategies_from_api(self, api_config):
        """Fetch strategies from an API source."""
        try:
            response = requests.get(api_config["url"], headers=api_config.get("headers", {}))
            response.raise_for_status()
            strategies = response.json()
            logger.log("info", f"Fetched {len(strategies)} strategies from API {api_config['name']}.")
            return strategies
        except requests.RequestException as e:
            logger.log("error", f"Error fetching strategies from API {api_config['name']}: {e}")
            handle_errors(e)
            return []

    def fetch_strategies_from_config(self):
        """Load strategies from a configuration file."""
        try:
            with open(self.config_file_path, 'r') as file:
                strategies = json.load(file)
                logger.log("info", f"Fetched {len(strategies)} strategies from config file.")
                return strategies
        except FileNotFoundError as e:
            logger.log("error", f"Config file not found: {e}")
            handle_errors(e)
            return []

# --- Flask Routes for Managing Sources ---
@app.route("/toggle_source", methods=["POST"])
def toggle_source():
    """Enable or disable a source dynamically."""
    data = request.json
    source_type = data["source_type"]
    source_name = data.get("source_name", None)
    enabled = data["enabled"]

    # Update source state
    if source_type == "database":
        strategy_manager.enabled_sources["database"] = enabled
    elif source_type == "config_file":
        strategy_manager.enabled_sources["config_file"] = enabled
    elif source_type == "apis" and source_name:
        for api in strategy_manager.api_configs:
            if api["name"] == source_name:
                api["enabled"] = enabled
                strategy_manager.enabled_sources["apis"].append(api)
                break
    else:
        return jsonify({"error": "Invalid source type or source name"}), 400

    return jsonify({"status": f"Source '{source_name or source_type}' set to {enabled}."})

@app.route("/add_api_source", methods=["POST"])
def add_api_source():
    """Add a new API source dynamically."""
    data = request.json
    api_config = {
        "name": data["name"],
        "url": data["url"],
        "headers": data.get("headers", {}),
        "enabled": True,
    }
    strategy_manager.api_configs.append(api_config)
    strategy_manager.enabled_sources["apis"].append(api_config)
    return jsonify({"status": f"API source '{data['name']}' added."})

@app.route("/delete_api_source", methods=["POST"])
def delete_api_source():
    """Delete an API source."""
    data = request.json
    api_name = data["name"]
    strategy_manager.api_configs = [
        api for api in strategy_manager.api_configs if api["name"] != api_name
    ]
    strategy_manager.enabled_sources["apis"] = [
        api for api in strategy_manager.enabled_sources["apis"] if api["name"] != api_name
    ]
    return jsonify({"status": f"API source '{api_name}' deleted."})

# Initialize the StrategyManager with example configs
db_config = {"host": "localhost", "user": "user", "password": "password", "database": "strategies_db"}
api_configs = [{"name": "API1", "url": "https://example.com/api1/strategies", "headers": {}, "enabled": True}]
config_file_path = "strategies_config.json"

strategy_manager = StrategyManager(db_config, api_configs, config_file_path)

if __name__ == "__main__":
    app.run(port=5000)
