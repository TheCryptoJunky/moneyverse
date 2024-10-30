import numpy as np
import logging

class Wallet:
    def __init__(self, initial_balance=0.0):
        """Initialize wallet with an initial balance."""
        self.balance = initial_balance
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Wallet initialized with balance: {self.balance}")

    def update_balance(self, amount):
        """Update balance by a specified amount, positive for deposit and negative for withdrawal."""
        if not isinstance(amount, (int, float)):
            self.logger.error("Invalid amount type for update_balance")
            raise ValueError("Amount must be a numeric value.")
        
        new_balance = self.balance + amount
        if new_balance < 0:
            self.logger.warning("Attempted to withdraw more than available balance.")
            raise ValueError("Insufficient funds for this transaction.")
        
        self.balance = new_balance
        self.logger.info(f"Updated balance by {amount}, new balance: {self.balance}")

    def get_balance(self):
        """Return the current balance."""
        self.logger.debug(f"Current balance retrieved: {self.balance}")
        return self.balance
