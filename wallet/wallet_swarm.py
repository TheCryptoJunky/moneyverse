import asyncio
import logging
from typing import List, Dict, Optional
from .wallet import Wallet
from ..strategies.mev_strategy import MEVStrategy
from ..managers.wallet_manager import WalletManager
from ..managers.strategy_manager import StrategyManager
from ..database.db_connection import DatabaseConnection

class WalletSwarm:
    """
    AI-enhanced swarm of wallets for autonomous, parallel MEV-based strategies.

    Attributes:
    - wallets (List[Wallet]): List of wallets in the swarm.
    - mev_strategy (MEVStrategy): Default MEV strategy for the swarm.
    - wallet_manager (WalletManager): Manages wallet assignments and tasks.
    - strategy_manager (StrategyManager): Manages and executes strategies.
    """

    def __init__(self, db: DatabaseConnection, wallet_addresses: Optional[List[str]] = None):
        """
        Initialize a new WalletSwarm instance with a central database connection.

        Args:
        - db (DatabaseConnection): Central database instance for logging and asset tracking.
        - wallet_addresses (Optional[List[str]]): Initial wallet addresses.
        """
        self.wallets = []
        self.wallet_addresses = wallet_addresses or []
        self.db = db
        self.wallet_manager = WalletManager(db)
        self.strategy_manager = StrategyManager(db)
        self.logger = logging.getLogger(__name__)

        for address in self.wallet_addresses:
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
        Balance assets across wallets for optimal distribution based on strategy needs.
        """
        total_value = sum(wallet.get_balance() for wallet in self.wallets)
        target_value = total_value / len(self.wallets)
        for wallet in self.wallets:
            current_balance = wallet.get_balance()
            difference = target_value - current_balance
            if abs(difference) > 0.01:
                self.wallet_manager.redistribute_assets(wallet, difference)
                self.logger.info(f"Balanced {wallet.address} by {difference}")

    async def execute_strategies(self):
        """
        Executes various strategies in parallel across the wallet swarm.
        """
        tasks = []
        for strategy_name in self.strategy_manager.get_all_strategy_names():
            for wallet in self.wallets:
                strategy_instance = self.strategy_manager.get_strategy(strategy_name)
                task = asyncio.create_task(strategy_instance.execute(wallet))
                tasks.append(task)
        await asyncio.gather(*tasks)
        self.logger.info("Executed strategies across the swarm.")

    def calculate_total_net_value(self) -> float:
        """
        Calculates and logs the total NAV for the swarm.
        """
        total_net_value = sum(wallet.calculate_net_value() for wallet in self.wallets)
        self.logger.info(f"Total Swarm NAV: {total_net_value}")
        return total_net_value

    async def run(self):
        """
        Continuously runs the wallet swarm, balancing assets and executing strategies.
        """
        while True:
            await self.balance_assets()
            await self.execute_strategies()
            await self.db.update_swarm_nav(self.calculate_total_net_value())
            await asyncio.sleep(1)  # Adjustable for timing

# Example usage
if __name__ == "__main__":
    db = DatabaseConnection()
    wallet_addresses = ["0xWalletAddress1", "0xWalletAddress2"]  # Example wallet addresses
    wallet_swarm = WalletSwarm(db, wallet_addresses)
    asyncio.run(wallet_swarm.run())
