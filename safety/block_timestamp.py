import logging
import time

class BlockTimestampChecker:
    """
    A class to check if a block is recent enough for reliable trading.
    Prevents interaction with stale or potentially manipulated blocks.
    """
    def __init__(self, max_block_age_seconds=30):
        self.max_block_age_seconds = max_block_age_seconds

    def is_block_fresh(self, block_timestamp):
        """Checks if the block's timestamp is within the acceptable range."""
        current_time = int(time.time())
        block_age = current_time - block_timestamp
        if block_age <= self.max_block_age_seconds:
            return True
        else:
            logging.warning(f"Block is too old. Age: {block_age} seconds.")
            return False
