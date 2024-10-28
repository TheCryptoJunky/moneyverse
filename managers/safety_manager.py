# File: /src/safety/safety_manager.py

from src.safety.poison_token_checker import PoisonTokenChecker
from src.safety.reorg_detection import ReorgDetection
from src.safety.circuit_breaker import CircuitBreaker

class SafetyManager:
    def __init__(self, db_connection):
        """
        Initializes the SafetyManager with all the safety checks (Poison Token Checker, Reorg Detection, Circuit Breaker).
        :param db_connection: The active database connection, if any safety checks depend on it.
        """
        self.db_connection = db_connection
        self.token_checker = PoisonTokenChecker()
        self.reorg_detection = ReorgDetection()
        self.circuit_breaker = CircuitBreaker()

    def check_token_safety(self, token_address):
        """
        Checks if the given token address is safe to trade with.
        Calls the PoisonTokenChecker to verify if the token is a scam or honeypot.

        :param token_address: The blockchain address of the token to be checked.
        :return: True if the token is safe to trade, False if it's a poison token.
        """
        if self.token_checker.is_poison_token(token_address):
            print(f"Token {token_address} is flagged as a poison token.")
            return False
        return True

    def check_for_reorgs(self):
        """
        Checks if a blockchain reorganization is detected.
        Calls ReorgDetection to verify blockchain consistency.

        :return: True if a reorg is detected, False if the chain is stable.
        """
        if self.reorg_detection.detect_reorg():
            print("Blockchain reorganization detected.")
            return True
        return False

    def check_circuit_breaker(self):
        """
        Checks if any market circuit breakers are triggered.
        Calls CircuitBreaker to check market conditions.

        :return: True if the market is safe for trading, False if circuit breaker is triggered.
        """
        if not self.circuit_breaker.is_market_safe():
            print("Market conditions are unsafe, circuit breaker triggered.")
            return False
        return True
