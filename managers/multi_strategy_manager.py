# moneyverse/managers/multi_strategy_manager.py

import logging
from typing import Dict, Any, List, Callable
from threading import Thread
import time

class MultiStrategyManager:
    """
    Manages multiple strategies in parallel, coordinating resource allocation, prioritization, and goal alignment.

    Attributes:
    - strategies (dict): Dictionary of strategy names mapped to strategy functions.
    - active_strategies (dict): Tracks active strategies and their statuses.
    - max_concurrent_strategies (int): Maximum number of strategies allowed to run simultaneously.
    - logger (Logger): Logs strategy execution and coordination.
    """

    def __init__(self, max_concurrent_strategies=3):
        self.strategies = {}  # {strategy_name: execute_func}
        self.active_strategies = {}  # {strategy_name: {"status": "running", "thread": Thread}}
        self.max_concurrent_strategies = max_concurrent_strategies
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MultiStrategyManager initialized with max concurrent strategies: {self.max_concurrent_strategies}")

    def register_strategy(self, strategy_name: str, execute_func: Callable):
        """
        Registers a strategy with its execution function.

        Args:
        - strategy_name (str): Name of the strategy.
        - execute_func (callable): Function that executes the strategy.
        """
        self.strategies[strategy_name] = execute_func
        self.logger.info(f"Registered strategy {strategy_name}.")

    def execute_strategy(self, strategy_name: str):
        """
        Executes a strategy in a separate thread if within the limit of concurrent strategies.

        Args:
        - strategy_name (str): Name of the strategy to execute.
        """
        if strategy_name in self.active_strategies:
            self.logger.warning(f"Strategy {strategy_name} is already running.")
            return

        if len(self.active_strategies) >= self.max_concurrent_strategies:
            self.logger.warning("Reached maximum concurrent strategies. Cannot execute more at this time.")
            return

        def run():
            self.logger.info(f"Starting execution of {strategy_name}")
            try:
                self.strategies[strategy_name]()  # Execute the strategy function
                self.logger.info(f"Completed execution of {strategy_name}")
            except Exception as e:
                self.logger.error(f"Error executing {strategy_name}: {str(e)}")
            finally:
                self.active_strategies.pop(strategy_name, None)

        strategy_thread = Thread(target=run)
        strategy_thread.start()
        self.active_strategies[strategy_name] = {"status": "running", "thread": strategy_thread}

    def monitor_strategies(self):
        """
        Continuously monitors active strategies, checking if they are still running.
        """
        self.logger.info("Starting strategy monitoring.")
        while True:
            for strategy_name, data in list(self.active_strategies.items()):
                if not data["thread"].is_alive():
                    self.logger.info(f"Strategy {strategy_name} has completed.")
                    self.active_strategies.pop(strategy_name)
            time.sleep(5)  # Adjustable monitoring interval

    def stop_strategy(self, strategy_name: str):
        """
        Stops a running strategy by terminating its thread (if possible).

        Args:
        - strategy_name (str): Name of the strategy to stop.
        """
        strategy = self.active_strategies.get(strategy_name)
        if strategy and strategy["thread"].is_alive():
            self.logger.info(f"Stopping strategy {strategy_name}")
            # Thread stopping is not natively supported; a cooperative exit flag would be needed
            # Placeholder for a cooperative termination mechanism
        else:
            self.logger.warning(f"Strategy {strategy_name} is not running or already completed.")

    def list_active_strategies(self) -> List[str]:
        """
        Lists all currently active strategies.

        Returns:
        - list: Names of active strategies.
        """
        active = list(self.active_strategies.keys())
        self.logger.info(f"Active strategies: {active}")
        return active

    def set_max_concurrent_strategies(self, max_strategies: int):
        """
        Sets the maximum number of concurrent strategies allowed to run.

        Args:
        - max_strategies (int): New maximum for concurrent strategies.
        """
        self.max_concurrent_strategies = max_strategies
        self.logger.info(f"Updated max concurrent strategies to {max_strategies}")
