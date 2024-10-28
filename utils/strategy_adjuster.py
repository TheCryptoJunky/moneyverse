import numpy as np
from typing import Dict

from .nav_monitor import NAVMonitor
from .mev_strategies import MEVStrategy

class StrategyAdjuster:
    def __init__(self, nav_monitor: NAVMonitor, mev_strategy: MEVStrategy):
        self.nav_monitor = nav_monitor
        self.mev_strategy = mev_strategy

    def adjust_strategy(self, current_timestamp: int, current_nav: float) -> Dict:
        adjustment_params = self.nav_monitor.adjust_strategy(current_timestamp, current_nav)
        # Update the MEV strategy parameters based on the adjustment params
        self.mev_strategy.update_params(adjustment_params)
        return adjustment_params
