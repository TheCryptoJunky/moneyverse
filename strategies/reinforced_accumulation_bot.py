# Full file path: /moneyverse/strategies/reinforced_accumulation_bot.py

import asyncio
import time
from statistics import mean
from exchange_interface import get_price, place_order
from centralized_logger import CentralizedLogger
from ai.rl_algorithms import RLAccumulationAgent  # RL agent for buy timing optimization
from utils import sentiment_analysis, fetch_historical_data, volatility_metrics
from database_manager import log_execution_data  # Log executions for RL analysis
from flask_gui import fetch_user_settings  # Fetch dashboard-controlled parameters

class ReinforcedAccumulationBot:
    def __init__(self, assets, budget=None):
        """
        Initialize the bot with multi-asset support, RL, and dynamic DCA/grid strategy.
        
        :param assets: List of dicts, each containing asset details (name, budget, dca_interval, buy_amount, etc.)
        :param budget: Total budget for all assets combined (optional).
        """
        self.assets = assets
        self.budget = budget
        self.total_spent = 0
        self.logger = CentralizedLogger()
        self.rl_agent = RLAccumulationAgent(environment="asset_accumulation")  # RL agent for optimized buy timing

    async def fetch_price(self, asset):
        """Fetches the latest price for a given asset."""
        try:
            price = await get_price(asset)
            self.logger.log_info(f"Fetched price for {asset}: {price}")
            return price
        except Exception as e:
            self.logger.log_error(f"Error fetching price for {asset}: {e}")
            return None

    async def place_accumulation_order(self, asset, amount, current_price):
        """Places an accumulation order and logs it for historical analysis."""
        try:
            await place_order(asset, 'buy', amount)
            self.total_spent += amount
            log_execution_data(asset, current_price, amount)  # Log data for RL refinement
            self.logger.log_info(f"Placed accumulation order for {amount} of {asset}")
        except Exception as e:
            self.logger.log_error(f"Error placing accumulation order: {e}")

    async def accumulate_asset(self, asset_config):
        """Run DCA and grid accumulation with dynamic RL-driven adjustments for a single asset."""
        asset = asset_config["name"]
        asset_budget = asset_config.get("budget", None)
        total_spent_on_asset = 0

        self.logger.log_info(f"Starting accumulation for {asset}")

        while True:
            # Fetch real-time settings from Flask GUI
            user_settings = fetch_user_settings(asset)
            dca_interval = user_settings.get("dca_interval", asset_config.get("dca_interval", 300))
            buy_amount = user_settings.get("buy_amount", asset_config.get("buy_amount", 1.0))
            grid_spacing = user_settings.get("grid_spacing", asset_config.get("grid_spacing", 0.02))
            dip_threshold = user_settings.get("dip_threshold", asset_config.get("dip_threshold", 0.02))

            # Check asset budget
            if asset_budget and total_spent_on_asset >= asset_budget:
                self.logger.log_info(f"Budget reached for {asset}. Stopping further purchases.")
                break

            # Fetch current price and analyze sentiment
            current_price = await self.fetch_price(asset)
            if not current_price:
                await asyncio.sleep(dca_interval)
                continue

            sentiment_score = sentiment_analysis(asset)
            rl_adjustment = self.rl_agent.decide_adjustment(current_price, sentiment_score)

            # Market condition analysis and scaling
            historical_data = fetch_historical_data(asset)  # Fetch historical data for RL agent
            volatility = volatility_metrics(historical_data)
            moving_avg = mean([data["price"] for data in historical_data[-30:]])  # Example 30-day average

            # Aggressiveness based on volatility and market condition
            if current_price < moving_avg and volatility < 0.05:  # Low volatility, stable market
                scaled_buy_amount = buy_amount * 1.5 * rl_adjustment["intensity"]
                dip_threshold *= 0.8  # More aggressive on stable conditions
            elif volatility > 0.2:  # High volatility
                scaled_buy_amount = buy_amount * 0.7 * rl_adjustment["intensity"]
                dip_threshold *= 1.2  # Conservative on volatile conditions
            else:
                scaled_buy_amount = buy_amount * rl_adjustment["intensity"]

            # Place DCA order
            await self.place_accumulation_order(asset, scaled_buy_amount, current_price)
            total_spent_on_asset += scaled_buy_amount

            # Grid Accumulation based on price intervals
            for i in range(1, 6):  # Grid with 5 intervals below current price
                grid_price = current_price * (1 - i * grid_spacing)
                if grid_price < current_price * (1 - dip_threshold):
                    grid_buy_amount = scaled_buy_amount * (1 + 0.1 * i)  # Increasing size in lower grids
                    self.logger.log_info(f"Grid buy triggered for {asset} at {grid_price}. Amount: {grid_buy_amount}")
                    await self.place_accumulation_order(asset, grid_buy_amount, grid_price)
                    total_spent_on_asset += grid_buy_amount

            # Global budget check
            if self.budget and self.total_spent >= self.budget:
                self.logger.log_info("Global budget reached. Halting all accumulation.")
                break

            await asyncio.sleep(dca_interval)  # Wait for next DCA cycle

    async def run(self):
        """Main loop for multi-asset accumulation with RL and dynamic adjustments based on GUI settings."""
        self.logger.log_info("Starting Reinforced Accumulation Bot with historical data analysis and GUI controls.")

        tasks = [self.accumulate_asset(asset_config) for asset_config in self.assets]
        await asyncio.gather(*tasks)  # Run all accumulation tasks in parallel

