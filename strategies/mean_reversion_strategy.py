# moneyverse/strategies/mean_reversion_strategy.py

import logging
import asyncio
from typing import Callable

class MeanReversionStrategyBot:
    """
    Executes a mean reversion strategy by identifying price deviations and trading towards the historical mean.

    Attributes:
    - price_monitor (Callable): Function to monitor asset prices for mean reversion conditions.
    - trade_executor (Callable): Function to execute trades when reversion conditions are met.
    - logger (Logger): Logs mean reversion actions and detected opportunities.
    """

    def __init__(self, price_monitor: Callable, trade_executor: Callable):
        self.price_monitor = price_monitor
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        self.logger.info("MeanReversionStrategyBot initialized.")

    async def monitor_prices(self):
        """
        Continuously monitors asset prices for mean reversion conditions.
        """
        self.logger.info("Monitoring prices for mean reversion conditions.")
        while True:
            opportunity = await self.price_monitor()
            if opportunity:
                await self.execute_mean_reversion_trade(opportunity)
            await asyncio.sleep(0.5)  # Interval for price checks

    async def execute_mean_reversion_trade(self, opportunity: dict):
        """
        Executes a trade based on mean reversion conditions.

        Args:
        - opportunity (dict): Data on detected price deviation.
        """
        asset = opportunity.get("asset")
        trade_side = opportunity.get("trade_side")  # "buy" or "sell"
        amount = opportunity.get("amount")
        self.logger.info(f"Executing mean reversion {trade_side} for {asset} with amount {amount}")

        # Execute the trade action based on mean reversion conditions
        success = await self.trade_executor(asset, trade_side, amount)
        if success:
            self.logger.info(f"Mean reversion trade succeeded for {asset}")
        else:
            self.logger.warning(f"Mean reversion trade failed for {asset}")

    # ---------------- Opportunity Handler for Mempool Integration Starts Here ----------------
    def handle_mean_reversion_opportunity(self, opportunity: dict):
        """
        Responds to detected mean reversion opportunities from MempoolMonitor.

        Args:
        - opportunity (dict): Opportunity data detected by the MempoolMonitor.
        """
        asset = opportunity.get("asset")
        trade_side = opportunity.get("trade_side")
        amount = opportunity.get("amount")

        self.logger.info(f"Mean reversion opportunity detected for {asset} with action {trade_side} and amount {amount}")

        # Execute mean reversion trade asynchronously
        asyncio.create_task(self.execute_mean_reversion_trade(opportunity))
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

        self.logger.info(f"Flash loan opportunity detected for mean reversion on {asset} with amount {amount}")
        asyncio.create_task(self.request_flash_loan(asset, amount))  # Trigger flash loan asynchronously
    # ---------------- Opportunity Handler for Flash Loan Integration Ends Here ----------------
