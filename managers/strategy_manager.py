import logging
from .arbitrage_strategy import ArbitrageStrategy
from .sandwich_strategy import SandwichStrategy
from .flash_loan_strategy import FlashLoanStrategy
from .liquidity_provision_strategy import LiquidityProvisionStrategy
from .statistical_arbitrage import StatisticalArbitrage
from .revenge_strategy import RevengeStrategy

class StrategyManager:
    """
    Manages and executes trading strategies based on dynamic market conditions.
    
    Attributes:
    - strategies (dict): Dictionary of available strategy instances.
    - logger (Logger): Logger for tracking strategy execution.
    """

    def __init__(self):
        self.strategies = {
            "arbitrage": ArbitrageStrategy(),
            "sandwich": SandwichStrategy(),
            "flash_loan": FlashLoanStrategy(),
            "liquidity_provision": LiquidityProvisionStrategy(),
            "statistical_arbitrage": StatisticalArbitrage(),
            "revenge": RevengeStrategy()
        }
        self.logger = logging.getLogger(__name__)
        self.logger.info("StrategyManager initialized with available strategies.")

    def select_strategy(self, condition: str):
        """
        Selects strategies based on current market condition.

        Args:
        - condition (str): Market condition like 'volatile', 'bullish', etc.

        Returns:
        - list: List of strategies appropriate for the condition.
        """
        selected_strategies = []
        if condition == "volatile":
            selected_strategies = [self.strategies["flash_loan"], self.strategies["sandwich"]]
        elif condition == "bullish":
            selected_strategies = [self.strategies["arbitrage"], self.strategies["liquidity_provision"]]
        elif condition == "neutral":
            selected_strategies = [self.strategies["statistical_arbitrage"], self.strategies["revenge"]]
        
        self.logger.info(f"Selected strategies for {condition} market: {[s.__class__.__name__ for s in selected_strategies]}")
        return selected_strategies

    def execute_strategy(self, wallet, strategy_name: str):
        """
        Executes a specified strategy for a given wallet.

        Args:
        - wallet (Wallet): Wallet instance for strategy execution.
        - strategy_name (str): Name of the strategy to execute.
        """
        strategy = self.strategies.get(strategy_name)
        if strategy:
            strategy.execute(wallet)
            self.logger.info(f"Executed {strategy_name} strategy for wallet {wallet.address}")
        else:
            self.logger.warning(f"Strategy {strategy_name} not found.")

    def execute_selected_strategies(self, wallets, condition: str):
        """
        Executes selected strategies on a list of wallets based on market conditions.

        Args:
        - wallets (list): List of Wallet objects.
        - condition (str): Market condition guiding strategy selection.
        """
        selected_strategies = self.select_strategy(condition)
        for strategy in selected_strategies:
            for wallet in wallets:
                strategy.execute(wallet)
                self.logger.info(f"Executed {strategy.__class__.__name__} on wallet {wallet.address}.")

    def log_strategy_performance(self, strategy_name: str, performance: dict):
        """
        Logs the performance of a strategy.

        Args:
        - strategy_name (str): Name of the strategy.
        - performance (dict): Performance metrics to log.
        """
        self.logger.info(f"Performance for {strategy_name}: {performance}")
