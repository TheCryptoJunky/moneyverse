# moneyverse/strategies/arbitrage_bot.py

import logging
from typing import Dict

class ArbitrageBot:
    """
    Executes arbitrage strategies across multiple markets and chains to capture price discrepancies.

    Attributes:
    - threshold (float): Minimum price difference required to trigger an arbitrage trade.
    - logger (Logger): Logs actions and detected opportunities.
    """

    def __init__(self, threshold=0.01):
        self.threshold = threshold  # e.g., 1% minimum profit margin
        self.logger = logging.getLogger(__name__)
        self.logger.info("ArbitrageBot initialized with threshold: {:.2%}".format(self.threshold))

    def detect_opportunity(self, market_data: Dict[str, float]) -> Dict[str, float]:
        """
        Identifies arbitrage opportunities based on market data.

        Args:
        - market_data (dict): Prices across different markets, keyed by market name.

        Returns:
        - dict: Contains buy and sell market info if an opportunity is detected.
        """
        markets = list(market_data.keys())
        opportunities = {}

        for i in range(len(markets)):
            for j in range(i + 1, len(markets)):
                buy_market, sell_market = markets[i], markets[j]
                buy_price, sell_price = market_data[buy_market], market_data[sell_market]

                if sell_price > buy_price * (1 + self.threshold):
                    opportunities = {'buy_market': buy_market, 'sell_market': sell_market, 'profit': sell_price - buy_price}
                    self.logger.info(f"Arbitrage detected: Buy from {buy_market}, sell to {sell_market} for profit of {opportunities['profit']}")
                    return opportunities
                
                elif buy_price > sell_price * (1 + self.threshold):
                    opportunities = {'buy_market': sell_market, 'sell_market': buy_market, 'profit': buy_price - sell_price}
                    self.logger.info(f"Arbitrage detected: Buy from {sell_market}, sell to {buy_market} for profit of {opportunities['profit']}")
                    return opportunities

        self.logger.debug("No arbitrage opportunities detected.")
        return opportunities

    def execute_arbitrage(self, wallet, opportunity: Dict[str, float], amount: float):
        """
        Executes an arbitrage by buying and selling across markets with a detected opportunity.

        Args:
        - wallet (Wallet): Wallet to use for the transaction.
        - opportunity (dict): Contains buy/sell markets and expected profit.
        - amount (float): Amount to trade.
        """
        if not opportunity:
            self.logger.warning("No opportunity available for execution.")
            return

        buy_market, sell_market = opportunity['buy_market'], opportunity['sell_market']
        profit = opportunity['profit']

        # Execute simulated trade by updating wallet balances
        if wallet.get_balance(buy_market) >= amount:
            wallet.update_balance(buy_market, -amount)
            wallet.update_balance(sell_market, amount * (1 + self.threshold))
            self.logger.info(f"Executed arbitrage: Bought {amount} on {buy_market}, sold on {sell_market}, for profit of {profit}.")
        else:
            self.logger.warning(f"Insufficient balance in {buy_market} for arbitrage.")

    def run(self, wallet, market_data: Dict[str, float], amount: float):
        """
        Detects and executes arbitrage opportunities in a specified wallet.

        Args:
        - wallet (Wallet): Wallet instance for execution.
        - market_data (dict): Prices across multiple markets.
        - amount (float): Amount to trade if an opportunity is detected.
        """
        opportunity = self.detect_opportunity(market_data)
        if opportunity:
            self.execute_arbitrage(wallet, opportunity, amount)
        else:
            self.logger.info("No arbitrage executed; no suitable opportunity detected.")
