# moneyverse/strategies/intra_exchange_arbitrage.py

import logging
import asyncio
from typing import Callable

class IntraExchangeArbitrageBot:
    """
    Executes intra-exchange arbitrage by identifying price discrepancies between asset pairs
    within a single exchange to capture profit from mispricings.

    Attributes:
    - intra_exchange_monitor (Callable): Function to monitor intra-exchange asset pairs for arbitrage.
    - trade_executor (Callable): Function to execute trades within the same exchange.
    - logger (Logger): Logs intra-exchange arbitrage actions and detected opportunities.
    """

    def __init__(self, intra_exchange_monitor: Callable, trade_executor: Callable):
        self.intra_exchange_monitor = intra_exchange_monitor
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        self.logger.info("IntraExchangeArbitrageBot initialized.")

    async def monitor_intra_exchange_opportunities(self):
        """
        Continuously monitors intra-exchange pairs for arbitrage opportunities.
        """
        self.logger.info("Monitoring intra-exchange pairs for arbitrage opportunities.")
        while True:
            intra_opportunity = await self.intra_exchange_monitor()
            if intra_opportunity:
                await self.execute_intra_exchange_trade(intra_opportunity)
            await asyncio.sleep(0.5)  # Frequent monitoring within the same exchange

    async def execute_intra_exchange_trade(self, intra_opportunity: dict):
        """
        Executes an intra-exchange arbitrage trade based on detected pair discrepancies.

        Args:
        - intra_opportunity (dict): Data on the detected intra-exchange price discrepancy.
        """
        asset_pair = intra_opportunity.get("asset_pair")
        buy_price = intra_opportunity.get("buy_price")
        sell_price = intra_opportunity.get("sell_price")
        amount = intra_opportunity.get("amount")
        self.logger.info(f"Executing intra-exchange arbitrage for {asset_pair} with amount {amount} between buy at {buy_price} and sell at {sell_price}")

        # Execute the intra-exchange arbitrage trade
        success = await self.trade_executor(asset_pair, buy_price, sell_price, amount)
        if success:
            self.logger.info(f"Intra-exchange arbitrage trade succeeded for {asset_pair}")
        else:
            self.logger.warning(f"Intra-exchange arbitrage trade failed for {asset_pair}")

    # ---------------- Opportunity Handler for Mempool Integration Starts Here ----------------
    def handle_intra_exchange_arbitrage_opportunity(self, intra_opportunity: dict):
        """
        Responds to detected intra-exchange arbitrage opportunities from MempoolMonitor.

        Args:
        - intra_opportunity (dict): Opportunity data detected by the MempoolMonitor.
        """
        asset_pair = intra_opportunity.get("asset_pair")
        buy_price = intra_opportunity.get("buy_price")
        sell_price = intra_opportunity.get("sell_price")
        amount = intra_opportunity.get("amount")

        self.logger.info(f"Intra-exchange arbitrage opportunity detected for {asset_pair} with amount {amount}")

        # Execute intra-exchange trade asynchronously
        asyncio.create_task(self.execute_intra_exchange_trade(intra_opportunity))
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

        self.logger.info(f"Flash loan opportunity detected for intra-exchange arbitrage on {asset} with amount {amount}")
        asyncio.create_task(self.request_flash_loan(asset, amount))  # Trigger flash loan asynchronously
    # ---------------- Opportunity Handler for Flash Loan Integration Ends Here ----------------
