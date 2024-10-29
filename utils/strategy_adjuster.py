# Full file path: moneyverse/utils/strategy_adjuster.py

import numpy as np
from typing import Dict
from centralized_logger import CentralizedLogger
from .nav_monitor import NAVMonitor
from .mev_strategies import MEVStrategy
from ai.rl_agent import RLTradingAgent

# Initialize a centralized logger
logger = CentralizedLogger()

class StrategyAdjuster:
    """
    Strategy Adjuster dynamically adjusts MEV strategy parameters based on NAV trends and AI-driven predictions.
    """

    def __init__(self, nav_monitor: NAVMonitor, mev_strategy: MEVStrategy, rl_agent: RLTradingAgent):
        """
        Initialize the StrategyAdjuster with NAVMonitor, MEVStrategy, and RL Agent.

        Args:
            nav_monitor (NAVMonitor): Monitors and tracks NAV for performance.
            mev_strategy (MEVStrategy): MEV strategy object for trading adjustments.
            rl_agent (RLTradingAgent): Reinforcement learning agent for strategy prediction.
        """
        self.nav_monitor = nav_monitor
        self.mev_strategy = mev_strategy
        self.rl_agent = rl_agent

    def adjust_strategy(self, current_timestamp: int, current_nav: float) -> Dict:
        """
        Adjusts the MEV strategy based on NAV performance and AI predictions.

        Args:
            current_timestamp (int): Current timestamp for reference.
            current_nav (float): Current NAV value to assess trends.

        Returns:
            Dict: Adjusted parameters for MEV strategy based on NAV trend and AI evaluation.
        """
        try:
            # Fetch adjustment parameters from NAVMonitor
            adjustment_params = self.nav_monitor.adjust_strategy(current_timestamp, current_nav)
            
            # Predictive adjustment using RL agent
            ai_prediction = self.rl_agent.predict_market_behavior(adjustment_params)
            adjustment_params["ai_adjustment_factor"] = ai_prediction  # AI-driven influence factor

            # Update MEV strategy parameters based on NAV performance and RL predictions
            self.mev_strategy.update_params(adjustment_params)
            logger.log("info", f"Strategy adjusted with parameters: {adjustment_params}")

            return adjustment_params

        except Exception as e:
            logger.log("error", f"Error adjusting strategy: {str(e)}")
            return {}

    def switch_strategy(self, market_conditions: Dict):
        """
        Switch to the most suitable strategy based on real-time market conditions.

        Args:
            market_conditions (Dict): Real-time metrics (e.g., volatility, liquidity).
        """
        try:
            # Define switching logic here, possibly using AI-driven insights from RL agent
            optimal_strategy = "MEV" if market_conditions["volatility"] > 0.7 else "Arbitrage"
            
            # Update active strategy in MEV strategy module or load a new strategy dynamically
            self.mev_strategy.set_active_strategy(optimal_strategy)
            logger.log("info", f"Switched to {optimal_strategy} strategy based on market conditions.")

        except Exception as e:
            logger.log("error", f"Error in strategy switching: {str(e)}")
