# Full file path: /moneyverse/strategies/market_maker_bot.py

import time
import numpy as np
from exchange_interface import get_order_book, place_order, cancel_order
from utils import sentiment_analysis, price_predictor  # AI-driven insights for strategy adjustments
from database_manager import reinvest_profits  # Reinvestment handling
from statistics import mean

class MarketMakerBot:
    def __init__(self, pair, spread=0.001, order_size=1.0, reinvest_percent=0.2, accumulation_wallet=None):
        self.pair = pair
        self.spread = spread
        self.order_size = order_size
        self.current_orders = []
        self.reinvest_percent = reinvest_percent
        self.accumulation_wallet = accumulation_wallet
        self.profit_tracker = []
        self.sentiment_threshold = 0.6  # Example threshold for sentiment-driven actions

    def analyze_market(self):
        """Fetches and returns the best bid and ask prices for market making."""
        order_book = get_order_book(self.pair)
        best_bid = order_book['bids'][0]
        best_ask = order_book['asks'][0]
        return best_bid, best_ask

    def adjust_strategy(self):
        """Adjusts spread and order size based on AI-driven sentiment and price prediction."""
        sentiment_score = sentiment_analysis(self.pair)
        price_prediction = price_predictor.predict_price_movement(self.pair)

        # Adjust spread based on sentiment
        if sentiment_score > self.sentiment_threshold:
            self.spread *= 0.8  # Narrow spread on positive sentiment
        else:
            self.spread *= 1.2  # Widen spread on negative sentiment

        # Modify order size based on predicted movement
        if price_prediction == 'up':
            self.order_size *= 1.5  # Increase buy size on positive predictions
        elif price_prediction == 'down':
            self.order_size *= 0.7  # Decrease buy size on negative predictions

    def adjust_orders(self):
        """Places buy and sell orders based on adjusted market data and spread."""
        best_bid, best_ask = self.analyze_market()
        target_bid = best_bid * (1 - self.spread)
        target_ask = best_ask * (1 + self.spread)

        self.cancel_current_orders()
        self.place_order('buy', target_bid, self.order_size)
        self.place_order('sell', target_ask, self.order_size)

    def place_order(self, side, price, size):
        """Places an order and tracks its ID for management."""
        order_id = place_order(self.pair, side, price, size)
        self.current_orders.append(order_id)

    def cancel_current_orders(self):
        """Cancels all current orders to prepare for new adjustments."""
        for order_id in self.current_orders:
            cancel_order(order_id)
        self.current_orders.clear()

    def reinvest_profits(self):
        """Calculates recent profits and reinvests into designated accumulation wallet."""
        if self.profit_tracker:
            recent_profit = mean(self.profit_tracker[-5:])  # Track recent profit
            reinvest_amount = recent_profit * self.reinvest_percent

            # Accumulate tokens in designated wallet if specified
            if self.accumulation_wallet:
                place_order(self.pair, 'buy', reinvest_amount, wallet=self.accumulation_wallet)
            else:
                place_order(self.pair, 'buy', reinvest_amount)  # Use default wallet if none specified

            # Reset profit tracker periodically to clear history and refine recent performance
            self.profit_tracker.clear()

    def run(self, interval=60):
        """Main bot loop to adjust strategy, place orders, and reinvest profits."""
        while True:
            self.adjust_strategy()  # AI-driven adjustments
            self.adjust_orders()
            time.sleep(interval)  # Delay for next market-making cycle
            self.reinvest_profits()  # Reinvest periodically to support liquidity pool
