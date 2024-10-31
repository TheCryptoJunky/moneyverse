# moneyverse/strategies/spread_arbitrage.py

import logging
import asyncio
from typing import Callable

class SpreadArbitrageBot:
    """
    Executes spread arbitrage by capitalizing on bid-ask spread differences for the same asset across one or multiple exchanges.

    Attributes:
    - spread_monitor (Callable): Function to monitor bid-ask spreads for arbitrage opportunities.
    - trade_executor (Callable): Function to execute trades based on spread opportunities.
    - logger (Logger): Logs spread arbitrage actions and detected opportunities.
    """

    def __init__(self, spread_monitor: Callable, trade_executor: Callable):
        self.spread_monitor = spread_monitor
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        self.logger.info("SpreadArbitrageBot initialized.")

    async def monitor_spread_opportunities(self):
        """
        Continuously monitors bid-ask spreads to detect arbitrage opportunities.
        """
        self.logger.info("Monitoring bid-ask spreads for arbitrage opportunities.")
        while True:
            spread_opportunity = await self.spread_monitor()
            if spread_opportunity:
                await self.execute_spread_trade(spread_opportunity)
            await asyncio.sleep(0.5)  # Adjust monitoring frequency as needed

    async def execute_spread_trade(self, spread_opportunity: dict):
        """
        Executes a trade based on a detected bid-ask spread discrepancy.

        Args:
        - spread_opportunity (dict): Data on the spread discrepancy for asset trading.
        """
        asset = spread_opportunity.get("asset")
        buy_price = spread_opportunity.get("buy_price")
        sell_price = spread_opportunity.get("sell_price")
        amount = spread_opportunity.get("amount")
        self.logger.info(f"Executing spread arbitrage for {asset} with amount {amount} at buy price {buy_price} and sell price {sell_price}")

        # Execute the spread trade
        success = await self.trade_executor(asset, buy_price, sell_price, amount)
        if success:
            self.logger.info("Spread arbitrage trade succeeded")
        else:
            self.logger.warning("Spread arbitrage trade failed")

    # ---------------- Opportunity Handler for Mempool Integration Starts Here ----------------
    def handle_spread_arbitrage_opportunity(self, spread_opportunity: dict):
        """
        Responds to detected spread arbitrage opportunities from MempoolMonitor.

        Args:
        - spread_opportunity (dict): Opportunity data detected by the MempoolMonitor.
        """
        asset = spread_opportunity.get("asset")
        buy_price = spread_opportunity.get("buy_price")
        sell_price = spread_opportunity.get("sell_price")
        amount = spread_opportunity.get("amount")

        self.logger.info(f"Spread arbitrage opportunity detected for {asset} with buy price {buy_price}, sell price {sell_price}, amount {amount}")

        # Execute spread arbitrage asynchronously
        asyncio.create_task(self.execute_spread_trade(spread_opportunity))
    # ---------------- Opportunity Handler Ends Here ----------------

    # ---------------- Opportunity Handler for Flash Loan Integration Starts Here ----------------
    async def request_and_execute_flash_loan(self, amount: float, spread_opportunity: dict):
        """
        Requests a flash loan and executes a spread arbitrage trade if the loan is granted.

        Args:
        - amount (float): The amount required for the trade.
        - spread_opportunity (dict): The trade opportunity to execute with the loan.
        """
        asset = spread_opportunity.get("asset")
        self.logger.info(f"Requesting flash loan of {amount} for spread arbitrage on {asset}")

        # Assume flash loan approval; execute spread trade with flash loan amount
        await self.execute_spread_trade(spread_opportunity)

    def handle_flash_loan_opportunity(self, opportunity: dict):
        """
        Responds to detected flash loan opportunities from FlashLoanMonitor.

        Args:
        - opportunity (dict): Opportunity data detected by FlashLoanMonitor.
        """
        asset = opportunity.get("asset")
        amount = opportunity.get("amount")
        spread_opportunity = opportunity.get("spread_opportunity", {})

        if spread_opportunity:
            self.logger.info(f"Flash loan opportunity detected for spread arbitrage on {asset} with amount {amount}")
            asyncio.create_task(self.request_and_execute_flash_loan(amount, spread_opportunity))
    # ---------------- Opportunity Handler for Flash Loan Integration Ends Here ----------------
