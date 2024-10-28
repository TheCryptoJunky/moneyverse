import numpy as np

class Wallet:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        
    def update_balance(self, amount):
        self.balance += amount
        
    def get_balance(self):
        return self.balance
