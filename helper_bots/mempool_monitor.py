# moneyverse/helper_bots/mempool_monitor.py

import logging
import time
from typing import Callable, List, Dict

class MempoolMonitor:
    """
    Centralized monitor for the mempool, notifying relevant strategies and managers
    when trade opportunities are detected.

    Attributes:
    - strategies (list): List of strategy functions to be notified on opportunities.
    - managers (list): List of manager functions to coordinate strategy actions.
    - analysis_func (callable): Function to analyze mempool transactions for opportunities.
    - logger (Logger): Logs monitoring activity, detections, and notifications.
    """

    def __init__(self, analysis_func: Callable, strategies: List[Callable] = None, managers: List[Callable] = None):
        self.analysis_func = analysis_func  # Analysis function from /utils/mempool_analysis.py
        self.strategies = strategies or []  # Strategies to notify (e.g., Sandwich Attack, Arbitrage)
        self.managers = managers or []  # Managers for coordination (e.g., Transaction Manager, Risk Manager)
        self.logger = logging.getLogger(__name__)
        self.logger.info("MempoolMonitor initialized for centralized monitoring.")

    def register_strategy(self, strategy_func: Callable):
        """
        Registers a strategy to be notified on detected opportunities.

        Args:
        - strategy_func (callable): Strategy function to be notified.
        """
        self.strategies.append(strategy_func)
        self.logger.info(f"Registered strategy {strategy_func.__name__} for mempool notifications.")

    def register_manager(self, manager_func: Callable):
        """
        Registers a manager to coordinate actions on detected opportunities.

        Args:
        - manager_func (callable): Manager function to be notified.
        """
        self.managers.append(manager_func)
        self.logger.info(f"Registered manager {manager_func.__name__} for mempool coordination.")

    def monitor_mempool(self, interval: float = 1.0):
        """
        Continuously monitors the mempool for trade opportunities.

        Args:
        - interval (float): Time interval between mempool checks in seconds.
        """
        self.logger.info("Starting mempool monitoring.")
        while True:
            opportunities = self.analysis_func()
            if opportunities:
                self.notify_strategies(opportunities)
                self.notify_managers(opportunities)
            time.sleep(interval)

    def notify_strategies(self, opportunities: List[Dict[str, any]]):
        """
        Notifies all registered strategies of detected opportunities.

        Args:
        - opportunities (list): List of opportunities detected in the mempool.
        """
        for strategy_func in self.strategies:
            for opportunity in opportunities:
                self.logger.info(f"Notifying {strategy_func.__name__} of opportunity in {opportunity['asset']}")
                strategy_func(opportunity)

    def notify_managers(self, opportunities: List[Dict[str, any]]):
        """
        Notifies all registered managers to coordinate strategy actions.

        Args:
        - opportunities (list): List of opportunities detected in the mempool.
        """
        for manager_func in self.managers:
            for opportunity in opportunities:
                self.logger.info(f"Notifying manager {manager_func.__name__} of opportunity in {opportunity['asset']}")
                manager_func(opportunity)
