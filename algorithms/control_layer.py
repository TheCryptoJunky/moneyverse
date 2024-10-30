import logging
from .managers.wallet_manager import WalletManager
from .managers.strategy_manager import StrategyManager
from .database.db_connection import DatabaseConnection
from .algorithms.ai_model import AIModel

class ControlLayer:
    """
    Central control layer for managing and monitoring bot operations.
    
    Attributes:
    - db (DatabaseConnection): Database for storing and retrieving bot data.
    - wallet_manager (WalletManager): Manages wallet assets and allocation.
    - strategy_manager (StrategyManager): Manages and executes trading strategies.
    - ai_model (AIModel): Selects and applies optimal models for trading decisions.
    """

    def __init__(self, db: DatabaseConnection, ai_model: AIModel):
        self.db = db
        self.wallet_manager = WalletManager(db)
        self.strategy_manager = StrategyManager()
        self.ai_model = ai_model
        self.logger = logging.getLogger(__name__)
        self.alert_thresholds = {"NAV_drop": 0.1, "high_volatility": 0.8}
        self.logger.info("ControlLayer initialized with wallet, strategy, and AI management.")

    def check_alerts(self):
        """
        Monitors critical metrics and triggers alerts if thresholds are breached.
        """
        nav = self.wallet_manager.calculate_total_nav()
        if nav < self.alert_thresholds["NAV_drop"]:
            self.logger.warning("NAV drop alert triggered.")
            self.db.log_alert("NAV drop alert")
        
        market_volatility = self.get_market_volatility()
        if market_volatility > self.alert_thresholds["high_volatility"]:
            self.logger.warning("High market volatility alert triggered.")
            self.db.log_alert("High volatility alert")

    def dynamic_allocation(self):
        """
        Adjusts bot allocation dynamically based on AI model recommendations.
        """
        market_condition = self.get_market_condition()
        model_output = self.ai_model.apply_best_model(market_condition, data=self.get_market_data())
        
        for wallet in self.wallet_manager.get_wallets():
            recommended_strategy = self.strategy_manager.get_strategy(model_output)
            self.strategy_manager.execute_prioritized_strategies([wallet])
            self.logger.info(f"Dynamically allocated strategy {recommended_strategy.__class__.__name__} to wallet {wallet.address}.")

    def manual_override(self, enable: bool):
        """
        Enables or disables manual override to directly control bot actions.
        
        Args:
        - enable (bool): If True, enables manual override.
        """
        if enable:
            self.logger.info("Manual override enabled; bots will follow manual input.")
        else:
            self.logger.info("Manual override disabled; bots will operate autonomously.")

    def get_market_condition(self) -> str:
        """
        Placeholder for retrieving current market condition (volatile, trending, neutral).
        
        Returns:
        - str: Market condition.
        """
        # Placeholder for actual market condition retrieval logic
        return "trending"

    def get_market_volatility(self) -> float:
        """
        Placeholder for retrieving current market volatility level.
        
        Returns:
        - float: Market volatility as a percentage.
        """
        # Placeholder for actual volatility retrieval logic
        return 0.5

    def get_market_data(self) -> dict:
        """
        Retrieves real-time market data for model input.

        Returns:
        - dict: Current market data.
        """
        # Placeholder for actual market data retrieval logic
        return {"market_trend": "bullish", "volatility": "low"}

    def run_control_layer(self):
        """
        Runs control layer, managing alerts, bot allocation, and overrides.
        """
        self.check_alerts()
        if not self.manual_override:
            self.dynamic_allocation()
        else:
            self.logger.info("Manual override is active; skipping dynamic allocation.")
