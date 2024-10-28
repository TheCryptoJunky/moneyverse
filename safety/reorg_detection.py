import logging

class ReorgDetector:
    """
    A class to detect blockchain reorganizations (reorgs).
    Ensures that trades are executed only in stable blocks not subject to reorgs.
    """
    def __init__(self, web3):
        self.web3 = web3
        self.last_block_hash = None

    def detect_reorg(self, block_number):
        """Detects if a reorg has occurred by comparing block hashes."""
        try:
            current_block = self.web3.eth.getBlock(block_number)
            if self.last_block_hash is None:
                self.last_block_hash = current_block.hash
                return False  # No reorg, first block processed

            # Compare the current block's parent hash to the last block hash
            if current_block.parentHash != self.last_block_hash:
                logging.error(f"Reorg detected at block {block_number}!")
                return True  # Reorg detected

            # Update the last block hash
            self.last_block_hash = current_block.hash
            return False  # No reorg detected
        except Exception as e:
            logging.error(f"Error detecting reorg: {e}")
            return False
