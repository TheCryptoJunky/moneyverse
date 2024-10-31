# moneyverse/helper_bots/price_oracle_bot.py

import logging
from typing import Dict, List

class PriceOracleBot:
    """
    Aggregates real-time price data across multiple exchanges or oracles, ensuring consistent and validated pricing.

    Attributes:
    - price_sources (dict): Dictionary of sources with functions to fetch price data.
    - price_data (dict): Latest price data aggregated from sources.
    - consistency_threshold (float): Allowable percentage difference between sources.
    - logger (Logger): Logs pricing updates and any detected inconsistencies.
    """

    def __init__(self, price_sources: Dict[str, callable], consistency_threshold=0.02):
        self.price_sources = price_sources  # {source_name: function to fetch price}
        self.consistency_threshold = consistency_threshold
        self.price_data = {}  # {asset: {source: price}}
        self.logger = logging.getLogger(__name__)
        self.logger.info("PriceOracleBot initialized with consistency threshold of {:.2%}".format(consistency_threshold))

    def fetch_prices(self, asset: str) -> Dict[str, float]:
        """
        Fetches price data from all sources for a specific asset.

        Args:
        - asset (str): Asset symbol to fetch price for.

        Returns:
        - dict: Dictionary of prices from each source.
        """
        prices = {}
        for source, fetch_func in self.price_sources.items():
            try:
                price = fetch_func(asset)
                prices[source] = price
                self.logger.debug(f"Fetched price for {asset} from {source}: {price}")
            except Exception as e:
                self.logger.error(f"Failed to fetch price from {source} for {asset}: {str(e)}")
        return prices

    def validate_price_consistency(self, prices: Dict[str, float]) -> bool:
        """
        Validates that prices from different sources are within the consistency threshold.

        Args:
        - prices (dict): Dictionary of price data from various sources.

        Returns:
        - bool: True if prices are consistent, False if discrepancies exceed the threshold.
        """
        if len(prices) < 2:
            return True  # No comparison needed if only one source

        avg_price = sum(prices.values()) / len(prices)
        for source, price in prices.items():
            if abs(price - avg_price) / avg_price > self.consistency_threshold:
                self.logger.warning(f"Price inconsistency detected for {source}: {price} deviates from avg {avg_price}")
                return False
        return True

    def update_price_data(self, asset: str):
        """
        Updates the aggregated price data for an asset and checks for consistency.

        Args:
        - asset (str): Asset symbol to update price data for.
        """
        prices = self.fetch_prices(asset)
        if self.validate_price_consistency(prices):
            self.price_data[asset] = prices
            avg_price = sum(prices.values()) / len(prices)
            self.logger.info(f"Updated consistent price for {asset}: {avg_price}")
        else:
            self.logger.warning(f"Inconsistent price data for {asset}: {prices}")

    def get_average_price(self, asset: str) -> float:
        """
        Returns the average price of an asset based on the latest data.

        Args:
        - asset (str): Asset symbol to retrieve average price for.

        Returns:
        - float: Average price across sources, or 0 if no data available.
        """
        if asset in self.price_data:
            avg_price = sum(self.price_data[asset].values()) / len(self.price_data[asset])
            self.logger.debug(f"Average price for {asset}: {avg_price}")
            return avg_price
        self.logger.warning(f"No price data available for {asset}.")
        return 0.0

    def check_price_alert(self, asset: str, alert_threshold: float):
        """
        Checks if price discrepancies exceed the alert threshold and logs an alert if so.

        Args:
        - asset (str): Asset symbol to check.
        - alert_threshold (float): Percentage threshold to trigger an alert.
        """
        if asset in self.price_data:
            prices = self.price_data[asset]
            avg_price = self.get_average_price(asset)
            for source, price in prices.items():
                if abs(price - avg_price) / avg_price > alert_threshold:
                    self.logger.warning(f"Price alert for {asset} on {source}: {price} deviates by >{alert_threshold:.2%}")
