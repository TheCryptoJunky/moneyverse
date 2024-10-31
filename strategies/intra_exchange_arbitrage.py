# moneyverse/strategies/intra_exchange_arbitrage.py

import logging
import asyncio
from typing import Callable

class IntraExchangeArbitrageBot:
    """
    Executes intra-exchange arbitrage by capitalizing on price differences across trading pairs within a single exchange.

    Attributes:
    - intra_exchange_monitor (Callable): Function to monitor price differences across pairs on the same exchange.
    - trade_executor (Callable): Function to execute trades based on intra-exchange arbitrage opportunities.
    - logger (Logger): Logs intra-exchange arbitrage actions and detected opportunities.
    """

    def __init__(self, intra_exchange_monitor: Callable, trade_executor: Callable):
        self.intra_exchange_monitor = intra_exchange_monitor
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        self.logger.info("IntraExchangeArbitrageBot initialized.")

    async def monitor_intra_exchange_opportunities(self):
        """
        Continuously monitors trading pairs within a single exchange to detect intra-exchange arbitrage opportunities.
        """
        self.logger.info("Monitoring trading pairs for intra-exchange arbitrage opportunities.")
        while True:
            intra_opportunity = await self.intra_exchange_monitor()
            if intra_opportunity:
                await self.execute_intra_exchange_trade(intra_opportunity)
            await asyncio.sleep(0.5)  # Adjust frequency as needed

    async def execute_intra_exchange_trade(self, intra_opportunity: dict):
        """
        Executes a trade based on a detected price discrepancy within an exchange.

        Args:
        - intra_opportunity (dict): Data on the price difference across pairs within an exchange.
        """
        asset_pair = intra_opportunity.get("asset_pair")
        amount = intra_opportunity.get("amount")
        buy_price = intra_opportunity.get("buy_price")
        sell_price = intra_opportunity.get("sell_price")
        self.logger.info(f"Executing intra-exchange arbitrage for {asset_pair} with amount {amount}")

        # Execute the trade based on intra-exchange discrepancy
        success = await self.trade_executor(asset_pair, amount, buy_price, sell_price)
        if success:
            self.logger.info("Intra-exchange arbitrage trade succeeded")
        else:
            self.logger.warning("Intra-exchange arbitrage trade failed")

    # ---------------- Opportunity Handler for Mempool Integration Starts Here ----------------
    def handle_intra_exchange_arbitrage_opportunity(self, intra_opportunity: dict):
        """
        Responds to detected intra-exchange arbitrage opportunities from MempoolMonitor.

        Args:
        - intra_opportunity (dict): Opportunity data detected by the MempoolMonitor.
        """
        asset_pair = intra_opportunity.get("asset_pair")
        amount = intra_opportunity.get("amount")

        self.logger.info(f"Intra-exchange arbitrage opportunity detected for {asset_pair} with amount {amount}")

        # Execute intra-exchange arbitrage asynchronously
        asyncio.create_task(self.execute_intra_exchange_trade(intra_opportunity))
    # ---------------- Opportunity Handler Ends Here ----------------

    # ---------------- Opportunity Handler for Flash Loan Integration Starts Here ----------------
    async def request_and_execute_flash_loan(self, amount: float, intra_opportunity: dict):
        """
        Requests a flash loan and executes an intra-exchange arbitrage trade if the loan is granted.

        Args:
        - amount (float): The amount required for the trade.
        - intra_opportunity (dict): The trade opportunity to execute with the loan.
        """
        asset_pair = intra_opportunity.get("asset_pair")
        self.logger.info(f"Requesting flash loan of {amount} for intra-exchange arbitrage on {asset_pair}")

        # Assume flash loan approval; execute intra-exchange trade with flash loan amount
        await self.execute_intra_exchange_trade(intra_opportunity)

    def handle_flash_loan_opportunity(self, opportunity: dict):
        """
        Responds to detected flash loan opportunities from FlashLoanMonitor.

        Args:
        - opportunity (dict): Opportunity data detected by FlashLoanMonitor.
        """
        asset_pair = opportunity.get("asset_pair")
        amount = opportunity.get("amount")
        intra_opportunity = opportunity.get("intra_opportunity", {})

        if intra_opportunity:
            self.logger.info(f"Flash loan opportunity detected for intra-exchange arbitrage on {asset_pair} with amount {amount}")
            asyncio.create_task(self.request_and_execute_flash_loan(amount, intra_opportunity))
    # ---------------- Opportunity Handler for Flash Loan Integration Ends Here ----------------
