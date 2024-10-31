# moneyverse/managers/safety_manager.py

import logging
from typing import Optional
from moneyverse.safety.block_finality import BlockFinalityChecker
from moneyverse.safety.reorg_detection import ReorgDetector
from moneyverse.safety.poison_token_checker import PoisonTokenChecker

class SafetyManager:
    """
    Manages security and operational safety by detecting potential risks such as chain reorgs, gas price spikes, and suspicious token activity.

    Attributes:
    - block_finality_checker (BlockFinalityChecker): Ensures finality of transactions on the blockchain.
    - reorg_detector (ReorgDetector): Detects chain reorganizations that could affect transaction validity.
    - poison_token_checker (PoisonTokenChecker): Identifies and flags potentially dangerous tokens.
    - logger (Logger): Logs safety actions and detected risks.
    """

    def __init__(self):
        self.block_finality_checker = BlockFinalityChecker()
        self.reorg_detector = ReorgDetector()
        self.poison_token_checker = PoisonTokenChecker()
        self.logger = logging.getLogger(__name__)
        self.logger.info("SafetyManager initialized with safety modules.")

    def check_block_finality(self, transaction_hash: str) -> bool:
        """
        Verifies if a transaction has reached finality on the blockchain.

        Args:
        - transaction_hash (str): Hash of the transaction to check.

        Returns:
        - bool: True if transaction is confirmed and final, False otherwise.
        """
        finality_status = self.block_finality_checker.is_final(transaction_hash)
        if not finality_status:
            self.logger.warning(f"Transaction {transaction_hash} has not reached finality.")
        return finality_status

    def detect_reorg(self) -> bool:
        """
        Checks for chain reorganization events that could affect transaction validity.

        Returns:
        - bool: True if reorganization is detected, False otherwise.
        """
        reorg_detected = self.reorg_detector.detect()
        if reorg_detected:
            self.logger.warning("Chain reorganization detected. Pausing operations.")
        return reorg_detected

    def check_for_poison_token(self, token_address: str) -> bool:
        """
        Checks if a token is flagged as suspicious or potentially dangerous.

        Args:
        - token_address (str): Address of the token to check.

        Returns:
        - bool: True if token is flagged, False otherwise.
        """
        is_poison = self.poison_token_checker.is_poisonous(token_address)
        if is_poison:
            self.logger.warning(f"Token at {token_address} is flagged as poisonous. Trading halted for this token.")
        return is_poison

    def enforce_safety_protocols(self, transaction_hash: Optional[str] = None, token_address: Optional[str] = None) -> bool:
        """
        Enforces all safety protocols, pausing operations if necessary.

        Args:
        - transaction_hash (str, optional): Transaction hash to verify finality.
        - token_address (str, optional): Token address to check for poison status.

        Returns:
        - bool: True if all protocols pass, False if any protocol flags an issue.
        """
        if transaction_hash and not self.check_block_finality(transaction_hash):
            self.logger.info("Finality check failed. Halting operations.")
            return False

        if self.detect_reorg():
            self.logger.info("Reorganization detected. Halting operations.")
            return False

        if token_address and self.check_for_poison_token(token_address):
            self.logger.info("Poison token detected. Halting operations.")
            return False

        self.logger.info("All safety protocols passed.")
        return True
