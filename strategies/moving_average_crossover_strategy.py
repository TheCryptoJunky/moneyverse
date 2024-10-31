# moneyverse/strategies/moving_average_crossover_strategy.py

import logging
import asyncio
from typing import Callable

class MovingAverageCrossoverStrategyBot:
    """
    Executes a moving average crossover strategy by entering trades when short-term moving averages cross long-term moving averages.

    Attributes:
    - ma_monitor (Callable): Function to monitor moving average conditions for crossovers.
    - trade_executor (Callable): Function to execute trades based on crossover conditions.
    - logger (Logger): Logs crossover actions and detected opportunities.
    """

    def __init__(self, ma_monitor: Callable, trade_executor: Callable):
        self.ma_monitor = ma_monitor
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        self.logger.info("MovingAverageCrossoverStrategyBot initialized.")

    async def monitor_moving_averages(self):
        """
        Continuously monitors moving averages for crossover conditions.
        """
        self.logger.info("Monitoring moving averages for crossover conditions.")
        while True:
            opportunity = await self.ma_monitor()
            if opportunity:
                await self.execute_crossover_trade(opportunity)
            await asyncio.sleep(0.5)  # Interval for frequent monitoring

    async def execute_crossover_trade(self, opportunity: dict):
        """
        Executes a trade based on detected moving average crossover.

        Args:
        - opportunity (dict): Data on detected moving average crossover.
        """
        asset = opportunity.get("asset")
        trade_side = opportunity.get("trade_side")  # "buy" or "sell"
        amount = opportunity.get("amount")
        self.logger.info(f"Executing moving average crossover {trade_side} for {asset} with amount {amount}")

        # Execute the trade action based on crossover conditions
        success = await self.trade_executor(asset, trade_side, amount)
        if success:
            self.logger.info(f"Moving average crossover trade succeeded for {asset}")
        else:
            self.logger.warning(f"Moving average crossover trade failed for {asset}")

    # ---------------- Opportunity Handler for Mempool Integration Starts Here ----------------
    def handle_moving_average_crossover_opportunity(self, opportunity: dict):
        """
        Responds to detected moving average crossover opportunities from MempoolMonitor.

        Args:
        - opportunity (dict): Opportunity data detected by the MempoolMonitor.
        """
        asset = opportunity.get("asset")
        trade_side = opportunity.get("trade_side")
        amount = opportunity.get("amount")

        self.logger.info(f"Moving average crossover opportunity detected for {asset} with action {trade_side} and amount {amount}")

        # Execute moving average crossover trade asynchronously
        asyncio.create_task(self.execute_crossover_trade(opportunity))
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

        self.logger.info(f"Flash loan opportunity detected for moving average crossover on {asset} with amount {amount}")
        asyncio.create_task(self.request_flash_loan(asset, amount))  # Trigger flash loan asynchronously
    # ---------------- Opportunity Handler for Flash Loan Integration Ends Here ----------------
