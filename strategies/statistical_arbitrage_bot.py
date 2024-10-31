# moneyverse/strategies/statistical_arbitrage_bot.py

import logging
import asyncio
from typing import Callable

class StatisticalArbitrageBot:
    """
    Executes statistical arbitrage by identifying price discrepancies between correlated assets.

    Attributes:
    - price_discrepancy_monitor (Callable): Function to monitor for price discrepancies.
    - arbitrage_executor (Callable): Function to execute arbitrage trades based on discrepancies.
    - logger (Logger): Logs arbitrage actions and detected opportunities.
    """

    def __init__(self, price_discrepancy_monitor: Callable, arbitrage_executor: Callable):
        self.price_discrepancy_monitor = price_discrepancy_monitor
        self.arbitrage_executor = arbitrage_executor
        self.logger = logging.getLogger(__name__)
        self.logger.info("StatisticalArbitrageBot initialized.")

    async def monitor_discrepancies(self):
        """
        Continuously monitors asset pairs for price discrepancies based on statistical correlations.
        """
        self.logger.info("Monitoring for statistical arbitrage opportunities.")
        while True:
            discrepancy = await self.price_discrepancy_monitor()
            if discrepancy:
                await self.execute_arbitrage(discrepancy)
            await asyncio.sleep(0.5)  # Interval for frequent checks

    async def execute_arbitrage(self, discrepancy: dict):
        """
        Executes an arbitrage trade based on detected price discrepancies.

        Args:
        - discrepancy (dict): Data on the detected price discrepancy.
        """
        asset_pair = discrepancy.get("asset_pair")
        buy_side = discrepancy.get("buy_side")  # "buy" or "sell"
        amount = discrepancy.get("amount")
        self.logger.info(f"Executing statistical arbitrage for {asset_pair} with {buy_side} side and amount {amount}")

        # Execute the arbitrage trade
        success = await self.arbitrage_executor(asset_pair, buy_side, amount)
        if success:
            self.logger.info(f"Statistical arbitrage trade succeeded for {asset_pair}")
        else:
            self.logger.warning(f"Statistical arbitrage trade failed for {asset_pair}")

    # ---------------- Opportunity Handler for Mempool Integration Starts Here ----------------
    def handle_statistical_arbitrage_opportunity(self, discrepancy: dict):
        """
        Responds to detected statistical arbitrage opportunities from MempoolMonitor.

        Args:
        - discrepancy (dict): Opportunity data detected by the MempoolMonitor.
        """
        asset_pair = discrepancy.get("asset_pair")
        buy_side = discrepancy.get("buy_side")
        amount = discrepancy.get("amount")

        self.logger.info(f"Statistical arbitrage opportunity detected for {asset_pair} with {buy_side} side and amount {amount}")

        # Execute arbitrage trade asynchronously
        asyncio.create_task(self.execute_arbitrage(discrepancy))
    # ---------------- Opportunity Handler Ends Here ----------------

    # ---------------- Opportunity Handler for Flash Loan Integration Starts Here ----------------
    def handle_flash_loan_opportunity(self, opportunity: dict):
        """
        Responds to detected flash loan opportunities from FlashLoanMonitor.

        Args:
        - opportunity (dict): Opportunity data detected by FlashLoanMonitor.
        """
        asset = opportunity.get("asset")
        amount = opportunity.get("amount")

        self.logger.info(f"Flash loan opportunity detected for statistical arbitrage on {asset} with amount {amount}")
        asyncio.create_task(self.request_flash_loan(asset, amount))  # Trigger flash loan asynchronously
    # ---------------- Opportunity Handler for Flash Loan Integration Ends Here ----------------
