import asyncio
import logging
from typing import List, Dict
from ..strategies import *
from ..algorithms.reinforcement_learning_agent import ReinforcementAgent

class StrategyManager:
    """
    StrategyManager dynamically manages and prioritizes trading strategies based on reinforcement feedback.
    
    Attributes:
    - strategies (Dict[str, MEVStrategy]): Active strategies available for execution.
    - reinforcement_agent (ReinforcementAgent): AI agent for strategy prioritization.
    """

    def __init__(self):
        self.strategies = {
            "arbitrage": ArbitrageBot(),
            "sandwich": EnhancedSandwichAttackBot(),
            "liquidity_drain": LiquidityDrainBot(),
            "latency_arbitrage": LatencyArbitrageBot(),
            # Add more strategies as necessary
        }
        self.reinforcement_agent = ReinforcementAgent()
        self.logger = logging.getLogger(__name__)

    def get_strategy(self, strategy_name: str):
        """
        Retrieves a strategy instance by name.
        
        Args:
        - strategy_name (str): Name of the strategy.
        
        Returns:
        - MEVStrategy: The strategy instance.
        """
        return self.strategies.get(strategy_name)

    def get_all_strategy_names(self) -> List[str]:
        """
        Returns all available strategy names.
        
        Returns:
        - list: Names of all active strategies.
        """
        return list(self.strategies.keys())

    async def execute_prioritized_strategies(self, wallets: List):
        """
        Executes prioritized strategies across multiple wallets based on real-time feedback from the reinforcement agent.
        
        Args:
        - wallets (List[Wallet]): List of wallet instances to run strategies on.
        """
        prioritized_strategies = self.reinforcement_agent.prioritize_strategies(self.get_all_strategy_names())
        tasks = []

        for strategy_name in prioritized_strategies:
            strategy_instance = self.get_strategy(strategy_name)
            for wallet in wallets:
                tasks.append(asyncio.create_task(self.run_strategy(strategy_instance, wallet)))
                
        await asyncio.gather(*tasks)
        self.logger.info("Executed prioritized strategies on all wallets.")

    async def run_strategy(self, strategy_instance, wallet):
        """
        Executes a single strategy instance on a wallet and updates the reinforcement agent.
        
        Args:
        - strategy_instance (MEVStrategy): Instance of the strategy to execute.
        - wallet (Wallet): Wallet on which to execute the strategy.
        """
        try:
            reward = await strategy_instance.execute(wallet)
            self.reinforcement_agent.update_strategy_performance(strategy_instance.__class__.__name__, reward)
            self.logger.info(f"Executed {strategy_instance.__class__.__name__} on wallet {wallet.address} with reward {reward}.")
        except Exception as e:
            self.logger.error(f"Error executing strategy {strategy_instance.__class__.__name__} on wallet {wallet.address}: {e}")
