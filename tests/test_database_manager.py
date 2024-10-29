# Full file path: /moneyverse/tests/test_database_manager.py

import unittest
from managers.database_manager import (
    add_pumplist_entry, remove_pumplist_entry, add_redlist_entry, log_performance_metric
)
from managers.database_manager import connect_db

class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        """Setup a fresh database connection and clean the test tables."""
        self.conn = connect_db()
        with self.conn:
            self.conn.execute("DELETE FROM pumplist")
            self.conn.execute("DELETE FROM redlist")
            self.conn.execute("DELETE FROM performance_metrics")

    def tearDown(self):
        """Close the database connection and clean up after each test."""
        with self.conn:
            self.conn.execute("DELETE FROM pumplist")
            self.conn.execute("DELETE FROM redlist")
            self.conn.execute("DELETE FROM performance_metrics")
        self.conn.close()

    def test_add_pumplist_entry(self):
        """Test adding a Pumplist entry to ensure it’s added with correct focus duration and strategies."""
        add_pumplist_entry("0xExampleToken", 30, ["market_maker", "accumulation"])
        
        # Verify entry in database
        cursor = self.conn.execute("SELECT * FROM pumplist WHERE entry_id = ?", ("0xExampleToken",))
        result = cursor.fetchone()
        self.assertIsNotNone(result, "Pumplist entry not found in database.")
        self.assertEqual(result["entry_id"], "0xExampleToken")
        self.assertEqual(result["focus_duration"], 30)
        self.assertIn("market_maker", result["strategies"])
        self.assertIn("accumulation", result["strategies"])

    def test_remove_pumplist_entry(self):
        """Test removing a Pumplist entry to ensure it’s deleted from the database."""
        add_pumplist_entry("0xExampleToken", 30, ["market_maker"])
        remove_pumplist_entry("0xExampleToken")
        
        # Verify entry is removed from database
        cursor = self.conn.execute("SELECT * FROM pumplist WHERE entry_id = ?", ("0xExampleToken",))
        result = cursor.fetchone()
        self.assertIsNone(result, "Pumplist entry was not removed from database.")

    def test_add_redlist_entry(self):
        """Test adding a Redlist entry to ensure it’s correctly logged in the database."""
        add_redlist_entry("0xBadActor")
        
        # Verify entry in Redlist
        cursor = self.conn.execute("SELECT * FROM redlist WHERE bad_actor = ?", ("0xBadActor",))
        result = cursor.fetchone()
        self.assertIsNotNone(result, "Redlist entry not found in database.")
        self.assertEqual(result["bad_actor"], "0xBadActor")

    def test_log_performance_metric(self):
        """Test logging a performance metric and verify it’s recorded in the performance_metrics table."""
        log_performance_metric("NAV Growth", 1.5)
        
        # Confirm entry in performance_metrics table
        cursor = self.conn.execute("SELECT * FROM performance_metrics WHERE metric_name = ?", ("NAV Growth",))
        result = cursor.fetchone()
        self.assertIsNotNone(result, "Performance metric not logged in database.")
        self.assertEqual(result["metric_name"], "NAV Growth")
        self.assertEqual(result["value"], 1.5)

if __name__ == "__main__":
    unittest.main()
