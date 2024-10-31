# moneyverse/managers/multi_strategy_manager.py

import logging
from typing import Dict, Any, List

class MultiStrategyManager:
    """
    Manages multiple strategies, coordinating their execution based on market conditions, goals, and potential conflicts.

    Attributes:
    - active_strategies (dict): Stores active strategies and their current status.
    - logger (Logger): Logs strategy activation, deactivation, and execution details.
    """

    def __init__(self):
        self.active_strategies = {}  # Dict of strategy names with their instances and statuses
        self.logger = logging.getLogger(__name__)
        self.logger.info("MultiStrategyManager initialized with empty strategy list.")

    def register_strategy(self, strategy_name: str, strategy_instance: Any):
        """
        Registers a new strategy by name.

        Args:
        - strategy_name (str): Name of the strategy.
        - strategy_instance (object): Instance of the strategy class.
        """
        self.active_strategies[strategy_name] = {
            "instance": strategy_instance,
            "status": "inactive"
        }
        self.logger.info(f"Registered strategy {strategy_name}.")

    def activate_strategy(self, strategy_name: str):
        """
        Activates a registered strategy for execution.

        Args:
        - strategy_name (str): Name of the strategy to activate.
        """
        if strategy_name in self.active_strategies:
            self.active_strategies[strategy_name]["status"] = "active"
            self.logger.info(f"Activated strategy {strategy_name}.")
        else:
            self.logger.warning(f"Strategy {strategy_name} not registered.")

    def deactivate_strategy(self, strategy_name: str):
        """
        Deactivates a registered strategy to prevent it from executing.

        Args:
        - strategy_name (str): Name of the strategy to deactivate.
        """
        if strategy_name in self.active_strategies:
            self.active_strategies[strategy_name]["status"] = "inactive"
            self.logger.info(f"Deactivated strategy {strategy_name}.")
        else:
            self.logger.warning(f"Strategy {strategy_name} not registered.")

    def run_active_strategies(self, wallet, market_data: Dict[str, float]):
        """
        Executes all active strategies with the provided market data.

        Args:
        - wallet (Wallet): Wallet instance to be used by strategies.
        - market_data (dict): Market data to be analyzed by strategies.
        """
        for strategy_name, strategy_info in self.active_strategies.items():
            if strategy_info["status"] == "active":
                strategy_instance = strategy_info["instance"]
                self.logger.info(f"Running active strategy {strategy_name}.")
                strategy_instance.run(wallet, market_data)

    def prioritize_strategies(self, priorities: List[str]):
        """
        Sets priority order for executing strategies.

        Args:
        - priorities (list): Ordered list of strategy names by priority.
        """
        ordered_strategies = {name: self.active_strategies[name] for name in priorities if name in self.active_strategies}
        self.active_strategies = {**ordered_strategies, **self.active_strategies}
        self.logger.info(f"Updated strategy execution order based on priorities: {priorities}")

    def get_active_strategies(self) -> List[str]:
        """
        Returns a list of currently active strategies.

        Returns:
        - list: Names of active strategies.
        """
        return [name for name, info in self.active_strategies.items() if info["status"] == "active"]
