import logging
from .managers.wallet_manager import WalletManager
from .managers.strategy_manager import StrategyManager
from .database.db_connection import DatabaseConnection
from .algorithms.ai_model import AIModel
from .algorithms.marl import MARLAgent

class ControlLayer:
    """
    Central control layer for managing and monitoring bot operations and dynamically reallocating resources.
    
    Attributes:
    - db (DatabaseConnection): Database connection for storing and retrieving bot data.
    - wallet_manager (WalletManager): Manages wallet assets and allocation.
    - strategy_manager (StrategyManager): Manages and executes trading strategies.
    - ai_model (AIModel): AI model selector for optimized trading decisions.
    - marl_agent (MARLAgent): Multi-Agent RL system for coordinated decision-making.
    """

    def __init__(self, db: DatabaseConnection, ai_model: AIModel, marl_agent: MARLAgent):
        self.db = db
        self.wallet_manager = WalletManager(db)
        self.strategy_manager = StrategyManager()
        self.ai_model = ai_model
        self.marl_agent = marl_agent
        self.logger = logging.getLogger(__name__)
        self.logger.info("ControlLayer initialized with wallet, strategy, AI model, and MARL management.")

    def evaluate_and_adjust_strategies(self):
        """
        Evaluates strategy performance and reallocates resources for optimal profitability.
        """
        strategies = self.strategy_manager.get_all_strategy_names()
        strategy_performance = {strategy: self.db.get_strategy_performance(strategy) for strategy in strategies}
        
        best_strategy = max(strategy_performance, key=strategy_performance.get)
        worst_strategy = min(strategy_performance, key=strategy_performance.get)

        if strategy_performance[best_strategy] > strategy_performance[worst_strategy] * 1.2:
            self.reallocate_resources(best_strategy, worst_strategy)
        
        self.logger.info(f"Strategies evaluated; best: {best_strategy}, worst: {worst_strategy}.")

    def reallocate_resources(self, best_strategy: str, worst_strategy: str):
        """
        Reallocates resources from underperforming to high-performing strategies.
        
        Args:
        - best_strategy (str): High-performing strategy.
        - worst_strategy (str): Underperforming strategy.
        """
        wallets = self.wallet_manager.get_wallets()
        selected_wallets = [wallet for wallet in wallets if wallet.strategy == worst_strategy]
        
        for wallet in selected_wallets:
            wallet.strategy = best_strategy
            self.logger.info(f"Reallocated wallet {wallet.address} from {worst_strategy} to {best_strategy}.")
        
        self.db.update_strategy_allocation(best_strategy, worst_strategy)
        self.logger.info(f"Resources reallocated to maximize profitability.")

    def monitor_and_adapt(self):
        """
        Monitors the system’s health and adapts resources and strategies in real-time.
        """
        self.evaluate_and_adjust_strategies()
        self.dynamic_allocation()

    def dynamic_allocation(self):
        """
        Adjusts bot allocation based on AI model recommendations and market data.
        """
        market_condition = self.get_market_condition()
        model_output = self.ai_model.apply_best_model(market_condition, data=self.get_market_data())
        
        for wallet in self.wallet_manager.get_wallets():
            recommended_strategy = self.strategy_manager.get_strategy(model_output)
            self.strategy_manager.execute_prioritized_strategies([wallet])
            self.logger.info(f"Dynamically allocated strategy {recommended_strategy.__class__.__name__} to wallet {wallet.address}.")

    def get_market_condition(self) -> str:
        """
        Placeholder for retrieving the current market condition (volatile, trending, neutral).
        
        Returns:
        - str: Market condition.
        """
        return "trending"

    def get_market_data(self) -> dict:
        """
        Retrieves real-time market data for model input.

        Returns:
        - dict: Current market data.
        """
        return {"market_trend": "bullish", "volatility": "low"}

    def run_control_layer(self):
        """
        Runs the control layer, managing strategy evaluation, resource allocation, and system adaptation.
        """
        self.monitor_and_adapt()
        if not self.manual_override:
            self.dynamic_allocation()
        else:
            self.logger.info("Manual override is active; skipping dynamic allocation.")
