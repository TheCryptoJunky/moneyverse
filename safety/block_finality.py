# File: /src/safety/block_finality.py

import time
import logging
from web3 import Web3

class BlockFinalityChecker:
    """
    A class to check block finality before allowing a trade.
    Ensures the block is deeply embedded in the chain (e.g., 12 confirmations).
    """

    def __init__(self, web3: Web3, confirmations_required=12):
        """
        Initializes the BlockFinalityChecker with the Web3 instance and required confirmations.

        Parameters:
        - web3 (Web3): An instance of Web3 to interact with the Ethereum network.
        - confirmations_required (int): Number of confirmations required to consider a block finalized.
        """
        self.web3 = web3
        self.confirmations_required = confirmations_required

    def is_block_finalized(self, block_number):
        """
        Checks if the block has enough confirmations to be considered finalized.

        Parameters:
        - block_number (int): The block number to check.

        Returns:
        - bool: True if the block is finalized, False otherwise.
        """
        try:
            latest_block = self.web3.eth.block_number
            confirmations = latest_block - block_number
            if confirmations >= self.confirmations_required:
                return True
            else:
                logging.warning(f"Block {block_number} has only {confirmations} confirmations.")
                return False
        except Exception as e:
            logging.error(f"Error checking block finality: {e}")
            return False

    def wait_for_finality(self, block_number):
        """
        Waits for the block to reach the required confirmations.

        Parameters:
        - block_number (int): The block number to wait for.
        """
        while not self.is_block_finalized(block_number):
            logging.info(f"Waiting for block {block_number} to be finalized...")
            time.sleep(15)  # Wait and recheck every 15 seconds
        logging.info(f"Block {block_number} is finalized.")
