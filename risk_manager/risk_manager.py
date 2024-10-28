# moneyverse/risk_manager.py

import numpy as np

class RiskManager:
    def __init__(self, threshold):
        self.threshold = threshold

    def assess_risk(self, portfolio_value):
        if portfolio_value < self.threshold:
            return True
        else:
            return False
