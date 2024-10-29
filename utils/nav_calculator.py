# Full file path: moneyverse/utils/nav_calculator.py

import numpy as np
from datetime import datetime, timedelta

class NAVCalculator:
    """
    A class to calculate Net Asset Value (NAV) for a wallet swarm with enhancements
    for caching, historical data storage, and multiple valuation methods.
    """
    def __init__(self, wallet_swarm, cache_duration=5):
        """
        Initialize NAVCalculator with the wallet swarm and optional caching.

        Args:
            wallet_swarm: The wallet swarm object containing all wallet data.
            cache_duration (int): Duration in minutes to cache NAV results for efficient recalculations.
        """
        self.wallet_swarm = wallet_swarm
        self.cache_duration = timedelta(minutes=cache_duration)
        self.nav_cache = None
        self.last_calculation_time = None

    def calculate_nav(self, method="default"):
        """
        Calculate the NAV for the wallet swarm using the specified method.
        
        Args:
            method (str): The calculation method to use ("default", "weighted", "moving_avg").
        
        Returns:
            float: The calculated NAV value.
        """
        current_time = datetime.now()
        
        # Check cache validity
        if self.nav_cache and self.last_calculation_time and \
           (current_time - self.last_calculation_time) < self.cache_duration:
            return self.nav_cache  # Return cached NAV if valid
        
        # Calculate NAV based on the selected method
        if method == "default":
            nav = self.wallet_swarm.calculate_nav()
        elif method == "weighted":
            nav = self._weighted_nav()
        elif method == "moving_avg":
            nav = self._moving_average_nav()
        else:
            raise ValueError(f"Unsupported calculation method: {method}")
        
        # Cache the calculated NAV and update calculation time
        self.nav_cache = nav
        self.last_calculation_time = current_time
        return nav

    def _weighted_nav(self):
        """
        Calculate NAV with a weighted valuation method.
        
        Returns:
            float: Weighted NAV value.
        """
        assets = self.wallet_swarm.get_assets()
        total_weight = sum(asset["weight"] for asset in assets)
        weighted_nav = sum(asset["value"] * asset["weight"] / total_weight for asset in assets)
        return weighted_nav

    def _moving_average_nav(self, period=5):
        """
        Calculate NAV based on a moving average of the last few valuations.
        
        Args:
            period (int): Number of past valuations to consider for moving average.
        
        Returns:
            float: Moving average NAV value.
        """
        history = self.wallet_swarm.get_nav_history()[-period:]  # Get last 'period' NAV entries
        moving_avg_nav = np.mean([entry["nav"] for entry in history]) if history else 0
        return moving_avg_nav

    def reset_cache(self):
        """Resets the NAV cache."""
        self.nav_cache = None
        self.last_calculation_time = None
