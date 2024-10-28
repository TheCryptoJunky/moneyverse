import unittest
import pandas as pd
from src.reports.performance_reporter import PerformanceReporter

class TestPerformanceReporting(unittest.TestCase):
    """Test suite for PerformanceReporter integration."""

    def setUp(self):
        """Set up mock transaction data for performance reporting."""
        # Create mock transaction data with necessary columns
        self.transaction_data = pd.DataFrame({
            'strategy': ['strategy_1', 'strategy_2'],
            'trades': [10, 15],  # Column required for groupby operation
            'profit': [1000, 2000],
            'win_rate': [0.75, 0.85]  # Added to avoid KeyError for 'win_rate'
        })

        # Initialize the PerformanceReporter with mock data
        self.reporter = PerformanceReporter(self.transaction_data)

    def test_generate_summary(self):
        """Test if the performance summary is generated correctly."""
        summary = self.reporter.generate_summary()
        self.assertIsNotNone(summary)

if __name__ == "__main__":
    unittest.main()
