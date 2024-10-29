# Full file path: moneyverse/utils/performance_tracker.py

import numpy as np
from datetime import datetime, timedelta
from collections import deque

class PerformanceTracker:
    """
    Tracks and analyzes the performance of the trading agent,
    recording NAV history and calculating various performance metrics.
    """
    
    def __init__(self, agent, history_period=30):
        """
        Initialize PerformanceTracker with an agent and history tracking.

        Args:
            agent: The trading agent to track.
            history_period (int): Number of past NAV values to store for historical analysis.
        """
        self.agent = agent
        self.history_period = history_period
        self.nav_history = deque(maxlen=history_period)  # Holds last `history_period` NAV values
        self.last_update_time = None

    def track_performance(self):
        """
        Tracks the agent's performance, specifically the Net Asset Value (NAV),
        and logs it for historical analysis.
        
        Returns:
            float: The current NAV calculated by the agent.
        """
        current_nav = self.agent.calculate_nav()
        self.nav_history.append({"timestamp": datetime.now(), "nav": current_nav})
        self.last_update_time = datetime.now()
        return current_nav

    def calculate_growth_rate(self):
        """
        Calculates the growth rate of NAV over the stored history period.
        
        Returns:
            float: The calculated growth rate (percent increase or decrease).
        """
        if len(self.nav_history) < 2:
            return 0  # Insufficient data to calculate growth rate

        initial_nav = self.nav_history[0]["nav"]
        final_nav = self.nav_history[-1]["nav"]
        growth_rate = ((final_nav - initial_nav) / initial_nav) * 100
        return growth_rate

    def get_performance_summary(self):
        """
        Provides a summary of performance metrics including current NAV,
        average NAV, and growth rate over the history period.
        
        Returns:
            dict: A dictionary containing performance metrics.
        """
        current_nav = self.nav_history[-1]["nav"] if self.nav_history else 0
        average_nav = np.mean([entry["nav"] for entry in self.nav_history]) if self.nav_history else 0
        growth_rate = self.calculate_growth_rate()
        
        return {
            "current_nav": current_nav,
            "average_nav": average_nav,
            "growth_rate": growth_rate,
            "last_update_time": self.last_update_time
        }

    def analyze_volatility(self):
        """
        Analyzes the volatility of NAV values by calculating the standard deviation.
        
        Returns:
            float: The standard deviation of NAV over the history period.
        """
        nav_values = [entry["nav"] for entry in self.nav_history]
        if len(nav_values) < 2:
            return 0  # Insufficient data for volatility calculation

        volatility = np.std(nav_values)
        return volatility

    def reset_tracker(self):
        """Resets the NAV history and last update time."""
        self.nav_history.clear()
        self.last_update_time = None
