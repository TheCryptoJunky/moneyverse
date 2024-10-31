# moneyverse/strategies/market_maker_bot.py

import logging
from typing import Dict

class MarketMakerBot:
    """
    Places buy and sell orders around the current market price to profit from the bid-ask spread.

    Attributes:
    - spread (float): Desired spread percentage between bid and ask prices.
    - order_size (float): Amount of asset to trade per order.
    - logger (Logger): Logs bot actions and detected opportunities.
    """

    def __init__(self, spread=0.005, order_size=10):
        self.spread = spread  # Spread percentage for bid and ask prices
        self.order_size = order_size  # Size of each order
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MarketMakerBot initialized with spread: {self.spread * 100}%, order size: {self.order_size}")

    def calculate_bid_ask_prices(self, market_price: float) -> Dict[str, float]:
        """
        Calculates bid and ask prices based on the market price and spread.

        Args:
        - market_price (float): Current market price of the asset.

        Returns:
        - dict: Contains calculated bid and ask prices.
        """
        bid_price = market_price * (1 - self.spread / 2)
        ask_price = market_price * (1 + self.spread / 2)
        self.logger.debug(f"Calculated bid price: {bid_price}, ask price: {ask_price}")
        return {'bid_price': bid_price, 'ask_price': ask_price}

    def place_orders(self, wallet, bid_price: float, ask_price: float):
        """
        Places bid and ask orders at calculated prices.

        Args:
        - wallet (Wallet): Wallet instance for placing orders.
        - bid_price (float): Calculated bid price for buying.
        - ask_price (float): Calculated ask price for selling.
        """
        # Simulate order placements by updating wallet balances for bid and ask
        wallet.update_balance("bid", self.order_size * bid_price)
        wallet.update_balance("ask", -self.order_size * ask_price)
        self.logger.info(f"Placed bid order at {bid_price} and ask order at {ask_price}.")

    def adjust_orders(self, wallet, current_market_price: float):
        """
        Adjusts existing orders based on new market conditions to maintain spread.

        Args:
        - wallet (Wallet): Wallet instance for updating orders.
        - current_market_price (float): Updated market price.
        """
        prices = self.calculate_bid_ask_prices(current_market_price)
        self.place_orders(wallet, prices['bid_price'], prices['ask_price'])
        self.logger.info("Adjusted orders to reflect new market conditions.")

    def run(self, wallet, market_price: float):
        """
        Executes market-making by placing and adjusting orders based on market movements.

        Args:
        - wallet (Wallet): Wallet instance for order placement.
        - market_price (float): Current market price for the asset.
        """
        prices = self.calculate_bid_ask_prices(market_price)
        self.place_orders(wallet, prices['bid_price'], prices['ask_price'])
        self.logger.info("Executed market-making strategy.")
