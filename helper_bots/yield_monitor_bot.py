# moneyverse/helper_bots/yield_monitor_bot.py

import logging
import asyncio
from typing import Callable, Dict

class YieldMonitorBot:
    """
    Continuously monitors yield rates across DeFi platforms and provides real-time data for other bots.

    Attributes:
    - yield_sources (Dict): Mapping of DeFi platforms to yield-fetching functions.
    - yield_data (Dict): Real-time yield data for each asset on different platforms.
    - logger (Logger): Logs monitoring actions and detected yield changes.
    """

    def __init__(self, yield_sources: Dict[str, Callable]):
        """
        Initialize YieldMonitorBot with yield-fetching functions for each platform.

        Args:
        - yield_sources (Dict[str, Callable]): Mapping of platform names to functions that fetch yield data.
        """
        self.yield_sources = yield_sources
        self.yield_data = {}  # { platform_name: { asset: yield_rate } }
        self.logger = logging.getLogger(__name__)
        self.logger.info("YieldMonitorBot initialized.")

    async def update_yield_rates(self):
        """
        Continuously fetches and updates yield rates from all specified platforms.
        """
        while True:
            for platform, fetch_yield in self.yield_sources.items():
                try:
                    yield_rates = await fetch_yield()  # Expected format: { asset: yield_rate }
                    self.yield_data[platform] = yield_rates
                    self.logger.info(f"Updated yield rates from {platform}: {yield_rates}")
                except Exception as e:
                    self.logger.error(f"Error fetching yield rates from {platform}: {e}")
            await asyncio.sleep(10)  # Set update frequency as needed

    def get_yield_rate(self, platform: str, asset: str) -> float:
        """
        Retrieves the current yield rate for a specified asset on a specified platform.

        Args:
        - platform (str): The DeFi platform name.
        - asset (str): The asset to check the yield rate for.

        Returns:
        - float: The yield rate if available, otherwise returns 0.0.
        """
        return self.yield_data.get(platform, {}).get(asset, 0.0)

    async def start_monitoring(self):
        """
        Starts the yield rate monitoring loop.
        """
        self.logger.info("Starting yield rate monitoring...")
        await self.update_yield_rates()

# Example usage and initialization
if __name__ == "__main__":
    async def fetch_yield_platform1():
        # Placeholder for fetching yield data from platform1
        return {"ETH": 0.05, "DAI": 0.03}  # Example yield rates

    async def fetch_yield_platform2():
        # Placeholder for fetching yield data from platform2
        return {"ETH": 0.04, "DAI": 0.035}  # Example yield rates

    # Initialize YieldMonitorBot with yield-fetching functions for different platforms
    yield_sources = {
        "Platform1": fetch_yield_platform1,
        "Platform2": fetch_yield_platform2,
    }

    yield_monitor_bot = YieldMonitorBot(yield_sources)
    asyncio.run(yield_monitor_bot.start_monitoring())
