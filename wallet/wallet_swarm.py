import asyncio
import logging
from typing import List
from .wallet import Wallet
from ..strategies.mev_strategy import MEVStrategy

class WalletSwarm:
    """
    Manages a swarm of wallets for obfuscation, resource distribution, and coordinated execution.

    Attributes:
    - wallets (List[Wallet]): List of wallets in the swarm.
    - mev_strategy (MEVStrategy): Strategy for maximized profit across wallets.
    - target_nav_growth (float): Targeted net asset value (NAV) growth rate.
    """

    def __init__(self, mev_strategy=None, wallet_addresses=None, target_nav_growth=2.0):
        self.wallets = []
        self.mev_strategy = mev_strategy
        self.wallet_addresses = wallet_addresses or []
        self.target_nav_growth = target_nav_growth  # Goal to double NAV
        self.logger = logging.getLogger(__name__)
        
        for address in self.wallet_addresses:
            self.add_wallet(address)

    def add_wallet(self, address, initial_balance=0.0):
        """
        Adds a wallet to the swarm.
        """
        wallet = Wallet(address=address, balance=initial_balance)
        self.wallets.append(wallet)
        self.logger.info(f"Added wallet {address} with balance {initial_balance}.")

    def remove_wallet(self, address):
        """
        Removes a wallet by its address.
        """
        self.wallets = [w for w in self.wallets if w.address != address]
        self.logger.info(f"Removed wallet {address} from the swarm.")

    def redistribute_assets(self):
        """
        Distributes assets among wallets to rebalance the swarm and meet target NAV.
        """
        total_balance = sum(wallet.balance for wallet in self.wallets)
        for wallet in self.wallets:
            wallet.balance = total_balance / len(self.wallets)
        self.logger.info(f"Redistributed assets. Each wallet balance updated to {total_balance / len(self.wallets)}.")

    async def execute_mev_strategy(self):
        """
        Executes the MEV strategy across all wallets asynchronously.
        """
        tasks = [self.mev_strategy.execute(wallet) for wallet in self.wallets]
        await asyncio.gather(*tasks)
        self.logger.info("MEV strategy executed for all wallets in swarm.")

    def calculate_total_nav(self) -> float:
        """
        Calculates the total NAV of the wallet swarm.
        """
        total_nav = sum(wallet.balance for wallet in self.wallets)
        self.logger.info(f"Total NAV of swarm calculated: {total_nav}")
        return total_nav

    def rebalance_swarm(self):
        """
        Rebalances the swarm based on performance and target NAV growth rate.
        """
        total_nav = self.calculate_total_nav()
        if total_nav < self.target_nav_growth:
            self.redistribute_assets()
            self.logger.info(f"Rebalanced swarm to align with target NAV growth: {self.target_nav_growth}")
        else:
            self.logger.info("Swarm NAV meets target; no rebalancing required.")

    async def run(self):
        """
        Runs the wallet swarm in a continuous loop, executing MEV strategies and rebalancing.
        """
        while True:
            await self.execute_mev_strategy()
            self.rebalance_swarm()
            await asyncio.sleep(1)  # Adjustable time interval for re-execution
