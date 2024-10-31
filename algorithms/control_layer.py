import logging
from .managers.wallet_manager import WalletManager
from .managers.strategy_manager import StrategyManager
from .database.db_connection import DatabaseConnection
from .algorithms.ai_model import AIModel
from .algorithms.marl import MARLAgent

class ControlLayer:
    """
    Central control layer for managing, monitoring, and dynamically adjusting trading strategies.
    
    Attributes:
    - db (DatabaseConnection): Database for logging performance and monitoring assets.
    - wallet_manager (WalletManager): Manages wallet balances and allocations.
    - strategy_manager (StrategyManager): Oversees execution and performance of trading strategies.
    - ai_model (AIModel): Selects and deploys optimized models based on conditions.
    - marl_agent (MARLAgent): Multi-agent reinforcement learning for collaborative strategy optimization.
    """

    def __init__(self, db: DatabaseConnection, ai_model: AIModel, marl_agent: MARLAgent):
        self.db = db
        self.wallet_manager = WalletManager(db)
        self.strategy_manager = StrategyManager()
        self.ai_model = ai_model
        self.marl_agent = marl_agent
        self.logger = logging.getLogger(__name__)

    def evaluate_strategies(self):
        """
        Evaluates each strategy's recent performance and reallocates resources accordingly.
        """
        strategies = self.strategy_manager.get_all_strategy_names()
        strategy_performance = {strategy: self.db.get_strategy_performance(strategy) for strategy in strategies}
        
        best_strategy = max(strategy_performance, key=strategy_performance.get)
        worst_strategy = min(strategy_performance, key=strategy_performance.get)
        
        if strategy_performance[best_strategy] > strategy_performance[worst_strategy] * 1.2:
            self.rebalance_resources(best_strategy, worst_strategy)
        
        self.logger.info(f"Strategies evaluated: best - {best_strategy}, worst - {worst_strategy}.")

    def rebalance_resources(self, best_strategy: str, worst_strategy: str):
        """
        Redistributes assets from underperforming to high-performing strategies.

        Args:
        - best_strategy (str): The top-performing strategy.
        - worst_strategy (str): The lowest-performing strategy.
        """
        wallets = self.wallet_manager.get_wallets()
        for wallet in wallets:
            if wallet.strategy == worst_strategy:
                wallet.strategy = best_strategy
                self.logger.info(f"Reallocated wallet {wallet.address} from {worst_strategy} to {best_strategy}.")
        self.db.update_strategy_allocation(best_strategy, worst_strategy)

    def dynamic_allocation(self):
        """
        Adjusts allocation based on AI model recommendations and risk assessment.
        """
        market_data = self.get_market_data()
        recommended_strategy = self.ai_model.apply_best_model(market_data["condition"], market_data)
        
        for wallet in self.wallet_manager.get_wallets():
            wallet.strategy = recommended_strategy
            self.strategy_manager.execute_strategy(wallet)
            self.logger.info(f"Applied strategy {recommended_strategy} to wallet {wallet.address}.")

    def assess_risk(self):
        """
        Conducts risk assessment based on market volatility and portfolio composition.
        """
        total_volatility = sum(self.db.get_asset_volatility(wallet.asset) for wallet in self.wallet_manager.get_wallets())
        risk_level = "high" if total_volatility > 1.5 else "medium" if total_volatility > 0.8 else "low"
        self.logger.info(f"Risk level assessed: {risk_level}")
        return risk_level

    def get_market_data(self) -> dict:
        """
        Placeholder function for retrieving real-time market data for model input.
        
        Returns:
        - dict: Contains condition and relevant market parameters.
        """
        return {"condition": "trending", "volatility": "low", "market_trend": "bullish"}

    def run_control_layer(self):
        """
        Manages strategy evaluation, risk assessment, and resource allocation.
        """
        self.evaluate_strategies()
        risk_level = self.assess_risk()
        
        if risk_level == "high":
            self.logger.info("High risk detected; rebalancing portfolio.")
            self.rebalance_resources("conservative", "aggressive")
        
        self.dynamic_allocation()
