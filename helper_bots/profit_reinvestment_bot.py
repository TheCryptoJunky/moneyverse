# moneyverse/helper_bots/profit_reinvestment_bot.py

import logging
from typing import Dict, List

class ProfitReinvestmentBot:
    """
    Reinvests profits into active strategies or assets based on performance metrics.

    Attributes:
    - reinvestment_threshold (float): Minimum profit amount required to trigger reinvestment.
    - allocation_strategy (dict): Distribution plan for reinvestment across strategies or assets.
    - logger (Logger): Logs reinvestment actions and decisions.
    """

    def __init__(self, reinvestment_threshold=0.01, allocation_strategy=None):
        self.reinvestment_threshold = reinvestment_threshold  # Minimum profit to reinvest
        self.allocation_strategy = allocation_strategy or {"arbitrage": 0.5, "trend_following": 0.5}
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ProfitReinvestmentBot initialized with threshold: {self.reinvestment_threshold} and allocation strategy: {self.allocation_strategy}")

    def calculate_reinvestment_amount(self, profit: float) -> float:
        """
        Calculates the amount to reinvest based on the profit and reinvestment threshold.

        Args:
        - profit (float): Profit made since the last reinvestment.

        Returns:
        - float: Amount to reinvest if it meets the threshold, 0 otherwise.
        """
        if profit >= self.reinvestment_threshold:
            self.logger.info(f"Calculated reinvestment amount: {profit}")
            return profit
        self.logger.debug(f"Profit of {profit} below threshold. No reinvestment triggered.")
        return 0.0

    def distribute_reinvestment(self, amount: float) -> Dict[str, float]:
        """
        Distributes the reinvestment amount across assets or strategies according to allocation strategy.

        Args:
        - amount (float): Total amount to reinvest.

        Returns:
        - dict: Dictionary with allocated amounts per strategy or asset.
        """
        allocations = {key: amount * ratio for key, ratio in self.allocation_strategy.items()}
        self.logger.info(f"Distributed reinvestment of {amount} as per allocation strategy: {allocations}")
        return allocations

    def reinvest(self, wallet, profit: float):
        """
        Reinvests the specified profit by allocating it across active strategies or assets.

        Args:
        - wallet (Wallet): Wallet instance to handle reinvestment.
        - profit (float): Total profit to reinvest.
        """
        reinvestment_amount = self.calculate_reinvestment_amount(profit)
        if reinvestment_amount > 0:
            allocations = self.distribute_reinvestment(reinvestment_amount)
            for strategy, amount in allocations.items():
                wallet.update_balance(strategy, amount)
                self.logger.info(f"Reinvested {amount} into {strategy} strategy.")
        else:
            self.logger.info("No reinvestment due to insufficient profit.")

    def adjust_allocation(self, new_allocation: Dict[str, float]):
        """
        Adjusts the allocation strategy for reinvestment.

        Args:
        - new_allocation (dict): New allocation percentages for each strategy.
        """
        if sum(new_allocation.values()) == 1.0:
            self.allocation_strategy = new_allocation
            self.logger.info(f"Adjusted allocation strategy: {self.allocation_strategy}")
        else:
            self.logger.warning("Allocation percentages must sum to 1.0. Adjustment not applied.")
