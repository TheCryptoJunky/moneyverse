import time
import threading
import logging
import gc  # Garbage collection

from config import Config  # Centralized configuration for API keys, etc.
from transaction_manager import TransactionManager  # To register transactions
from strategy_manager import StrategyManager  # For strategy execution
from centralized_logger import CentralizedLogger  # For centralized logging

# Initialize logger
logger = logging.getLogger(__name__)
centralized_logger = CentralizedLogger()

class MultiAgentManager:
    def __init__(self, config, strategy_manager, transaction_manager):
        self.config = config
        self.strategy_manager = strategy_manager
        self.transaction_manager = transaction_manager
        self.bots = {}  # Dictionary to hold all active bots
        self.stop_flags = {}  # Flags to safely stop bots
        self.threads = {}  # To track running bot threads

    def add_bot(self, bot_id, bot):
        """Adds a new bot and runs it in a separate thread."""
        if bot_id in self.bots:
            logger.warning(f"Bot with ID {bot_id} already exists.")
            return

        self.bots[bot_id] = bot
        self.stop_flags[bot_id] = False
        bot_thread = threading.Thread(target=self._run_bot, args=(bot_id,))
        bot_thread.start()
        self.threads[bot_id] = bot_thread
        logger.info(f"Bot {bot_id} added and started.")

    def _run_bot(self, bot_id):
        """Runs the bot in a loop until the stop flag is set."""
        bot = self.bots.get(bot_id)
        if not bot:
            logger.error(f"Bot with ID {bot_id} not found.")
            return

        while not self.stop_flags[bot_id]:
            try:
                bot.run()
            except Exception as e:
                logger.error(f"Error running bot {bot_id}: {e}")
            time.sleep(1)  # Adjust based on bot frequency
            gc.collect()

    def stop_bot(self, bot_id):
        """Stops the bot by setting the stop flag and joining the thread."""
        if bot_id not in self.bots:
            logger.error(f"Bot with ID {bot_id} not found.")
            return

        self.stop_flags[bot_id] = True
        self.threads[bot_id].join()  # Wait for thread to finish
        del self.bots[bot_id]
        del self.threads[bot_id]
        del self.stop_flags[bot_id]
        logger.info(f"Bot {bot_id} stopped and removed.")

    def stop_all_bots(self):
        """Gracefully stops all bots."""
        logger.info("Stopping all bots...")
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)

    def restart_bot(self, bot_id):
        """Restarts a bot by stopping it and starting it again."""
        if bot_id not in self.bots:
            logger.error(f"Bot with ID {bot_id} not found.")
            return

        logger.info(f"Restarting bot {bot_id}...")
        self.stop_bot(bot_id)
        bot = self.strategy_manager.get_strategy(bot_id)  # Retrieve the bot again
        self.add_bot(bot_id, bot)

    def get_active_bots(self):
        """Returns a list of active bots."""
        return list(self.bots.keys())
