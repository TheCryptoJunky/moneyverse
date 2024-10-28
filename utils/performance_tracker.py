import numpy as np

class PerformanceTracker:
    def __init__(self, agent):
        self.agent = agent

    def track_performance(self):
        # Track performance using agent
        performance = self.agent.calculate_nav()
        return performance
