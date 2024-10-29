# Full file path: moneyverse/utils/reward_calculator.py

class RewardCalculator:
    """
    Calculates rewards for trades based on profitability, risk, and other trading metrics.
    Supports multiple reward strategies to optimize reinforcement learning.
    """
    
    def __init__(self, profit_weight=1.0, risk_penalty=0.5, consistency_bonus=0.3):
        """
        Initialize RewardCalculator with configurable weights for different reward factors.
        
        Args:
            profit_weight (float): Weight for the profit factor in the reward calculation.
            risk_penalty (float): Penalty weight for taking on excessive risk.
            consistency_bonus (float): Bonus for achieving consistent trades.
        """
        self.profit_weight = profit_weight
        self.risk_penalty = risk_penalty
        self.consistency_bonus = consistency_bonus

    def calculate_reward(self, trade):
        """
        Calculates the reward for a given trade based on profit, risk, and consistency.
        
        Args:
            trade (dict): Trade data containing 'profit', 'risk', and 'consistency' values.
        
        Returns:
            float: Calculated reward for the trade.
        """
        profit_reward = self._calculate_profit_reward(trade.get("profit", 0))
        risk_penalty = self._calculate_risk_penalty(trade.get("risk", 0))
        consistency_bonus = self._calculate_consistency_bonus(trade.get("consistency", 0))
        
        reward = profit_reward - risk_penalty + consistency_bonus
        return max(reward, 0)  # Ensures reward does not go below zero

    def _calculate_profit_reward(self, profit):
        """
        Calculates the reward component based on profit.
        
        Args:
            profit (float): Profit value from the trade.
        
        Returns:
            float: Profit reward component.
        """
        return profit * self.profit_weight

    def _calculate_risk_penalty(self, risk):
        """
        Calculates the penalty component based on risk.
        
        Args:
            risk (float): Risk metric for the trade.
        
        Returns:
            float: Risk penalty component.
        """
        return risk * self.risk_penalty

    def _calculate_consistency_bonus(self, consistency):
        """
        Calculates the bonus component for consistency.
        
        Args:
            consistency (float): Consistency metric for the trade.
        
        Returns:
            float: Consistency bonus component.
        """
        return consistency * self.consistency_bonus
