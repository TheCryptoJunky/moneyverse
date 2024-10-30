import asyncio
import logging
from typing import List, Dict, Optional
from .wallet import Wallet
from ..strategies.mev_strategy import MEVStrategy
from ..managers.wallet_manager import WalletManager
from ..managers.strategy_manager import StrategyManager
from ..database.db_connection import DatabaseConnection
from ..algorithms.reinforcement_agent import ReinforcementAgent

class WalletSwarm:
    """
    AI-enhanced swarm of wallets for autonomous, parallel MEV-based strategies.

    Attributes:
    - wallets (List[Wallet]): List of wallets in the swarm.
    - strategy_manager (StrategyManager): Manages and coordinates strategies.
    - wallet_manager (WalletManager): Manages wallet assignments and tasks.
    - reinforcement_agent (ReinforcementAgent): Core RL agent for strategy adjustments.
    """

    def __init__(self, db: DatabaseConnection, wallet_addresses: Optional[List[str]] = None):
        """
        Initialize a new WalletSwarm instance with a central database connection.

        Args:
        - db (DatabaseConnection): Database instance for logging and tracking.
        - wallet_addresses (Optional[List[str]]): List of wallet addresses.
        """
        self.wallets = []
        self.db = db
        self.strategy_manager = StrategyManager(db)
        self.wallet_manager = WalletManager(db)
        self.reinforcement_agent = ReinforcementAgent()
        self.logger = logging.getLogger(__name__)

        for address in wallet_addresses or []:
            self.add_wallet(address)

    def add_wallet(self, address: str, initial_balance: float = 0.0):
        """
        Add a new wallet to the swarm with an initial balance.
        """
        wallet = Wallet(address=address, initial_balance=initial_balance)
        self.wallets.append(wallet)
        self.wallet_manager.register_wallet(wallet)
        self.logger.info(f"Added wallet {address} with initial balance {initial_balance}.")

    async def balance_assets(self):
        """
        Balance assets dynamically across wallets to meet trading needs and maximize NAV growth.
        """
        total_value = sum(wallet.get_balance() for wallet in self.wallets)
        target_value = total_value / len(self.wallets)
        for wallet in self.wallets:
            balance_diff = target_value - wallet.get_balance()
            if abs(balance_diff) > 0.01:
                self.wallet_manager.redistribute_assets(wallet, balance_diff)
                self.logger.info(f"Balanced wallet {wallet.address} by {balance_diff}.")

    async def execute_strategies(self):
        """
        Executes all strategies in parallel based on real-time data and RL-based prioritization.
        """
        tasks = []
        prioritized_strategies = self.reinforcement_agent.prioritize_strategies(
            self.strategy_manager.get_all_strategy_names()
        )
        for strategy_name in prioritized_strategies:
            for wallet in self.wallets:
                strategy_instance = self.strategy_manager.get_strategy(strategy_name)
                tasks.append(asyncio.create_task(strategy_instance.execute(wallet)))
        await asyncio.gather(*tasks)
        self.logger.info("Executed prioritized strategies across the swarm.")

    def calculate_total_nav(self) -> float:
        """
        Calculates and logs the total NAV for the wallet swarm.
        """
        total_nav = sum(wallet.calculate_net_value() for wallet in self.wallets)
        self.logger.info(f"Total Swarm NAV: {total_nav}")
        return total_nav

    async def run(self):
        """
        Initializes and continuously runs the wallet swarm, rebalancing assets and executing strategies.
        """
        while True:
            await self.balance_assets()
            await self.execute_strategies()
            await self.db.update_swarm_nav(self.calculate_total_nav())
            await asyncio.sleep(1)  # Adjustable interval

# Example usage
if __name__ == "__main__":
    db = DatabaseConnection()
    wallet_addresses = ["0xWalletAddress1", "0xWalletAddress2"]  # Example addresses
    wallet_swarm = WalletSwarm(db, wallet_addresses)
    asyncio.run(wallet_swarm.run())
