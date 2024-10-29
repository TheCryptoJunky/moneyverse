# Full file path: /moneyverse/tests/test_centralized_logger.py

import unittest
import os
from all_logging.centralized_logger import CentralizedLogger

class TestCentralizedLogger(unittest.TestCase):
    log_file = "test_log.log"

    def setUp(self):
        """Set up the test environment by initializing the logger and ensuring a clean log file."""
        self.logger = CentralizedLogger(log_file=self.log_file)
        # Remove existing log file if it exists to start fresh
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def tearDown(self):
        """Clean up by removing the test log file."""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_log_info(self):
        """Test logging an info message and verify it is written to the log file."""
        test_message = "Test info message"
        self.logger.log_info(test_message)

        # Verify message is in the log file
        with open(self.log_file, "r") as log:
            log_content = log.read()
            self.assertIn(test_message, log_content, "Info message was not logged correctly.")

    def test_log_event(self):
        """Test logging a structured event message and verify content and format in the log file."""
        event_type = "ADD_TO_PUMPLIST"
        entry_id = "0xExampleToken"
        description = "Added to Pumplist for testing"
        expected_message = f"{event_type} | Entry ID: {entry_id} | {description}"

        self.logger.log_event(event_type, entry_id, description)

        # Verify the event message structure is in the log file
        with open(self.log_file, "r") as log:
            log_content = log.read()
            self.assertIn(expected_message, log_content, "Event message was not logged correctly.")

if __name__ == "__main__":
    unittest.main()
