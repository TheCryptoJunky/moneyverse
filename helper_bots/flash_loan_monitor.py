# moneyverse/helper_bots/flash_loan_monitor.py

import logging
import time
from typing import Callable, List, Dict

class FlashLoanMonitor:
    """
    Monitors liquidity pools and markets for flash loan arbitrage opportunities.

    Attributes:
    - analysis_func (Callable): Function to analyze pools for flash loan opportunities.
    - strategies (List[Callable]): List of strategy functions to be notified on opportunities.
    - managers (List[Callable]): List of manager functions to coordinate strategy actions.
    - logger (Logger): Logs monitoring, detections, and notifications.
    """

    def __init__(self, analysis_func: Callable, strategies: List[Callable] = None, managers: List[Callable] = None):
        self.analysis_func = analysis_func  # Analysis function for flash loan conditions
        self.strategies = strategies or []  # Strategies to notify (e.g., Flash Loan Arbitrage)
        self.managers = managers or []  # Managers for coordination (e.g., Risk Manager, Transaction Manager)
        self.logger = logging.getLogger(__name__)
        self.logger.info("FlashLoanMonitor initialized for centralized flash loan monitoring.")

    def register_strategy(self, strategy_func: Callable):
        """
        Registers a strategy to be notified on detected flash loan opportunities.
        """
        self.strategies.append(strategy_func)
        self.logger.info(f"Registered strategy {strategy_func.__name__} for flash loan notifications.")

    def register_manager(self, manager_func: Callable):
        """
        Registers a manager to coordinate actions on detected flash loan opportunities.
        """
        self.managers.append(manager_func)
        self.logger.info(f"Registered manager {manager_func.__name__} for flash loan coordination.")

    def monitor_flash_loans(self, interval: float = 1.0):
        """
        Continuously monitors for flash loan arbitrage opportunities.

        Args:
        - interval (float): Time interval between checks in seconds.
        """
        self.logger.info("Starting flash loan monitoring.")
        while True:
            opportunities = self.analysis_func()
            if opportunities:
                self.notify_strategies(opportunities)
                self.notify_managers(opportunities)
            time.sleep(interval)

    def notify_strategies(self, opportunities: List[Dict[str, any]]):
        """
        Notifies all registered strategies of detected opportunities.
        """
        for strategy_func in self.strategies:
            for opportunity in opportunities:
                self.logger.info(f"Notifying {strategy_func.__name__} of flash loan opportunity for {opportunity['asset']}")
                strategy_func(opportunity)

    def notify_managers(self, opportunities: List[Dict[str, any]]):
        """
        Notifies all registered managers to coordinate strategy actions.
        """
        for manager_func in self.managers:
            for opportunity in opportunities:
                self.logger.info(f"Notifying manager {manager_func.__name__} of flash loan opportunity for {opportunity['asset']}")
                manager_func(opportunity)
