import pandas as pd

class PerformanceReporter:
    """Generates performance reports for trading strategies."""

    def __init__(self, transaction_data: pd.DataFrame):
        """Initialize with transaction data."""
        if not isinstance(transaction_data, pd.DataFrame):
            raise ValueError("transaction_data must be a pandas DataFrame.")
        self.transaction_data = transaction_data

    def generate_summary(self):
        """Generate a summary of the performance data."""
        summary = self.transaction_data.groupby('strategy').agg({
            'trades': 'sum',
            'profit': 'sum',
            'win_rate': 'mean'
        })
        return summary
