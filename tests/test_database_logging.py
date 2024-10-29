# Full file path: /moneyverse/tests/test_database_logging.py

import unittest
from managers.database_manager import add_pumplist_entry, remove_pumplist_entry, add_redlist_entry, fetch_pumplist_entry, fetch_redlist_entry

class TestDatabaseLogging(unittest.TestCase):

    def setUp(self):
        """Set up test data if needed."""
        # Add initial setup code if necessary

    def tearDown(self):
        """Cleanup test data to ensure no persistence between tests."""
        remove_pumplist_entry("0xExampleToken")
        remove_pumplist_entry("0xTestToken")
        remove_redlist_entry("0xBadActor")

    def test_add_pumplist_logging(self):
        """Test adding an entry to the pumplist and validate insertion."""
        result = add_pumplist_entry("0xExampleToken", 30, ["accumulation"])
        self.assertIsNotNone(result, "Failed to add Pumplist entry to the database.")
        
        # Fetch entry directly to confirm it was added
        entry = fetch_pumplist_entry("0xExampleToken")
        self.assertIsNotNone(entry, "Entry was not found in the pumplist after insertion.")

    def test_remove_pumplist_logging(self):
        """Test removing an entry from the pumplist and validate deletion."""
        add_pumplist_entry("0xTestToken", 45, ["market_maker"])
        
        # Ensure entry exists before removal
        self.assertIsNotNone(fetch_pumplist_entry("0xTestToken"), "Entry not found in pumplist before removal.")
        
        # Remove entry and verify removal
        remove_pumplist_entry("0xTestToken")
        result = fetch_pumplist_entry("0xTestToken")
        self.assertIsNone(result, "Failed to remove Pumplist entry from the database.")

    def test_add_redlist_logging(self):
        """Test adding an entry to the redlist and validate insertion."""
        result = add_redlist_entry("0xBadActor")
        self.assertIsNotNone(result, "Failed to log Redlist entry in the database.")
        
        # Fetch entry directly to confirm it was added
        entry = fetch_redlist_entry("0xBadActor")
        self.assertIsNotNone(entry, "Redlist entry was not found in the database after insertion.")

if __name__ == "__main__":
    unittest.main()
