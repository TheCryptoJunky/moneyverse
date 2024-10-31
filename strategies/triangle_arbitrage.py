# moneyverse/strategies/triangle_arbitrage_bot.py

import logging
import asyncio
from typing import Callable

class TriangleArbitrageBot:
    """
    Executes triangle arbitrage by exploiting price discrepancies between three assets in a triangular trading loop.

    Attributes:
    - price_monitor (Callable): Function to monitor asset prices for triangular arbitrage conditions.
    - trade_executor (Callable): Function to execute trades based on identified triangular discrepancies.
    - logger (Logger): Logs triangular arbitrage actions and detected opportunities.
    """

    def __init__(self, price_monitor: Callable, trade_executor: Callable):
        self.price_monitor = price_monitor
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        self.logger.info("TriangleArbitrageBot initialized.")

    async def monitor_triangular_arbitrage(self):
        """
        Continuously monitors prices for triangular arbitrage conditions.
        """
        self.logger.info("Monitoring for triangular arbitrage opportunities.")
        while True:
            opportunity = await self.price_monitor()
            if opportunity:
                await self.execute_triangle_arbitrage(opportunity)
            await asyncio.sleep(0.5)  # Adjusted for frequent checks

    async def execute_triangle_arbitrage(self, opportunity: dict):
        """
        Executes a triangular arbitrage trade based on detected opportunity.

        Args:
        - opportunity (dict): Data on the triangular price discrepancy.
        """
        assets = opportunity.get("assets")  # Tuple of the three assets involved in the loop
        trade_sequence = opportunity.get("trade_sequence")  # Sequence of trades in the loop
        amount = opportunity.get("amount")
        self.logger.info(f"Executing triangular arbitrage on {assets} with trade sequence {trade_sequence} and amount {amount}")

        # Execute the triangular arbitrage trade
        success = await self.trade_executor(assets, trade_sequence, amount)
        if success:
            self.logger.info(f"Triangular arbitrage trade succeeded for assets {assets}")
        else:
            self.logger.warning(f"Triangular arbitrage trade failed for assets {assets}")

    # ---------------- Opportunity Handler for Mempool Integration Starts Here ----------------
    def handle_triangle_arbitrage_opportunity(self, opportunity: dict):
        """
        Responds to detected triangular arbitrage opportunities from MempoolMonitor.

        Args:
        - opportunity (dict): Opportunity data detected by the MempoolMonitor.
        """
        assets = opportunity.get("assets")
        trade_sequence = opportunity.get("trade_sequence")
        amount = opportunity.get("amount")

        self.logger.info(f"Triangular arbitrage opportunity detected for assets {assets} with sequence {trade_sequence} and amount {amount}")

        # Execute triangular arbitrage asynchronously
        asyncio.create_task(self.execute_triangle_arbitrage(opportunity))
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

        self.logger.info(f"Flash loan opportunity detected for triangular arbitrage on {asset} with amount {amount}")
        asyncio.create_task(self.request_flash_loan(asset, amount))  # Trigger flash loan asynchronously
    # ---------------- Opportunity Handler for Flash Loan Integration Ends Here ----------------
