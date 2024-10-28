# Full file path: /moneyverse/managers/safety_manager.py

from src.safety.poison_token_checker import PoisonTokenChecker
from src.safety.reorg_detection import ReorgDetection
from src.safety.circuit_breaker import CircuitBreaker
import logging

# Set up centralized logging
logger = logging.getLogger(__name__)

class SafetyManager:
    """
    Manages safety checks for trading operations, including poison token detection, reorg monitoring, and circuit breakers.
    """

    def __init__(self, db_connection=None):
        """
        Initialize SafetyManager with essential safety checks.
        
        :param db_connection: Optional database connection if required by certain safety checks.
        """
        self.db_connection = db_connection
        self.token_checker = PoisonTokenChecker()
        self.reorg_detection = ReorgDetection()
        self.circuit_breaker = CircuitBreaker()

    def check_token_safety(self, token_address):
        """
        Check if a given token is safe for trading.
        
        :param token_address: The blockchain address of the token.
        :return: True if safe, False if identified as a poison token.
        """
        try:
            is_poison = self.token_checker.is_poison_token(token_address)
            if is_poison:
                logger.warning(f"Token {token_address} flagged as a poison token.")
                return False
            logger.info(f"Token {token_address} passed safety check.")
            return True
        except Exception as e:
            logger.error(f"Error in token safety check for {token_address}: {e}")
            return False

    def check_for_reorgs(self):
        """
        Check for blockchain reorganizations.
        
        :return: True if a reorg is detected, False otherwise.
        """
        try:
            if self.reorg_detection.detect_reorg():
                logger.warning("Blockchain reorganization detected.")
                return True
            logger.info("No blockchain reorganization detected.")
            return False
        except Exception as e:
            logger.error(f"Error in reorg detection: {e}")
            return False

    def check_circuit_breaker(self):
        """
        Check if the market circuit breaker is triggered.
        
        :return: True if the market is safe, False if conditions are unsafe.
        """
        try:
            if not self.circuit_breaker.is_market_safe():
                logger.warning("Circuit breaker triggered due to unsafe market conditions.")
                return False
            logger.info("Market conditions are safe.")
            return True
        except Exception as e:
            logger.error(f"Error in circuit breaker check: {e}")
            return False
