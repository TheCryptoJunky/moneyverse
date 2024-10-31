# moneyverse/managers/bot_manager.py

import logging
from typing import Dict, Any
from moneyverse.helper_bots import HelperBot  # Import base class or interfaces as needed

class BotManager:
    """
    Manages lifecycle and performance monitoring for each bot in the system.

    Attributes:
    - active_bots (dict): Dictionary storing active bot instances with their statuses.
    - logger (Logger): Logs lifecycle events and performance of bots.
    """

    def __init__(self):
        self.active_bots = {}  # Stores bot instances with lifecycle statuses
        self.logger = logging.getLogger(__name__)
        self.logger.info("BotManager initialized with empty bot list.")

    def register_bot(self, bot_name: str, bot_instance: HelperBot):
        """
        Registers a new bot to the manager.

        Args:
        - bot_name (str): Name of the bot to register.
        - bot_instance (HelperBot): Instance of the bot to manage.
        """
        self.active_bots[bot_name] = {
            "instance": bot_instance,
            "status": "stopped"
        }
        self.logger.info(f"Registered bot {bot_name}.")

    def start_bot(self, bot_name: str):
        """
        Starts a registered bot and updates its status.

        Args:
        - bot_name (str): Name of the bot to start.
        """
        if bot_name in self.active_bots and self.active_bots[bot_name]["status"] != "running":
            bot_instance = self.active_bots[bot_name]["instance"]
            bot_instance.start()
            self.active_bots[bot_name]["status"] = "running"
            self.logger.info(f"Started bot {bot_name}.")
        else:
            self.logger.warning(f"Bot {bot_name} already running or not registered.")

    def stop_bot(self, bot_name: str):
        """
        Stops a running bot and updates its status.

        Args:
        - bot_name (str): Name of the bot to stop.
        """
        if bot_name in self.active_bots and self.active_bots[bot_name]["status"] == "running":
            bot_instance = self.active_bots[bot_name]["instance"]
            bot_instance.stop()
            self.active_bots[bot_name]["status"] = "stopped"
            self.logger.info(f"Stopped bot {bot_name}.")
        else:
            self.logger.warning(f"Bot {bot_name} is not running or not registered.")

    def monitor_bot_performance(self, bot_name: str) -> Dict[str, Any]:
        """
        Monitors and logs the performance metrics of a running bot.

        Args:
        - bot_name (str): Name of the bot to monitor.

        Returns:
        - dict: Performance metrics of the bot.
        """
        if bot_name in self.active_bots and self.active_bots[bot_name]["status"] == "running":
            bot_instance = self.active_bots[bot_name]["instance"]
            metrics = bot_instance.get_performance_metrics()
            self.logger.info(f"Performance metrics for {bot_name}: {metrics}")
            return metrics
        else:
            self.logger.warning(f"Bot {bot_name} is not running or not registered.")
            return {}

    def restart_bot(self, bot_name: str):
        """
        Restarts a bot by stopping and starting it again.

        Args:
        - bot_name (str): Name of the bot to restart.
        """
        self.stop_bot(bot_name)
        self.start_bot(bot_name)
        self.logger.info(f"Restarted bot {bot_name}.")

    def list_active_bots(self) -> Dict[str, str]:
        """
        Lists all registered bots with their current statuses.

        Returns:
        - dict: Dictionary of bot names and their statuses.
        """
        self.logger.info("Listing all active bots and their statuses.")
        return {bot_name: info["status"] for bot_name, info in self.active_bots.items()}
