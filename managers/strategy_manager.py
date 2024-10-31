# moneyverse/managers/strategy_manager.py

import logging
from moneyverse.strategies.arbitrage_bot import ArbitrageBot
from moneyverse.strategies.flash_loan_arbitrage_bot import FlashLoanArbitrageBot
from moneyverse.strategies.front_running_bot import FrontRunningBot
from moneyverse.strategies.mean_reversion_strategy import MeanReversionStrategy
from moneyverse.safety.risk_manager import RiskManager
from moneyverse.database.db_connection import DatabaseConnection

class StrategyManager:
    """
    Centralized manager to control and execute various trading strategies.

    Attributes:
    - strategies (dict): Collection of available strategies, each initialized once.
    - risk_manager (RiskManager): Ensures safety checks are passed before strategy execution.
    - db (DatabaseConnection): Database connection for logging and performance tracking.
    """

    def __init__(self, db: DatabaseConnection):
        self.logger = logging.getLogger(__name__)
        self.strategies = {
            "arbitrage": ArbitrageBot(),
            "flash_loan_arbitrage": FlashLoanArbitrageBot(),
            "front_running": FrontRunningBot(),
            "mean_reversion": MeanReversionStrategy()
            # Add additional strategies as necessary
        }
        self.risk_manager = RiskManager(db)
        self.db = db
        self.logger.info("StrategyManager initialized with available strategies.")

    def execute_strategy(self, strategy_name: str, **kwargs):
        """
        Executes a specified strategy after performing safety checks.

        Args:
        - strategy_name (str): Name of the strategy to execute.
        - **kwargs: Additional parameters specific to the strategy.
        """
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            self.logger.error(f"Strategy {strategy_name} not found.")
            return
        
        # Perform risk checks before execution
        if not self.risk_manager.is_safe_to_execute(strategy_name):
            self.logger.warning(f"Risk Manager blocked execution for {strategy_name}")
            return

        # Execute strategy and log completion
        self.logger.info(f"Executing {strategy_name} strategy.")
        strategy.run(**kwargs)
        self.logger.info(f"Completed execution of {strategy_name} strategy.")
    
    def execute_all_strategies(self, **kwargs):
        """
        Executes all available strategies in parallel, after individual safety checks.

        Args:
        - **kwargs: Parameters passed to each strategy's execution method.
        """
        for strategy_name in self.strategies.keys():
            self.execute_strategy(strategy_name, **kwargs)

    def stop_all_strategies(self):
        """
        Gracefully stops all running strategies, if they have a stop method.
        """
        for strategy in self.strategies.values():
            if hasattr(strategy, "stop"):
                strategy.stop()
        self.logger.info("Stopped all strategies.")
