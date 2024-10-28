import unittest
from unittest.mock import Mock
from src.ai.momentum_strategy import MomentumStrategy

class TestMomentumStrategy(unittest.TestCase):
    """Unit tests for the MomentumStrategy."""

    def setUp(self):
        """Set up the momentum strategy and mock trading environment."""
        self.mock_trading_env = Mock()
        self.momentum_strategy = MomentumStrategy(self.mock_trading_env)

    def test_buy_signal(self):
        """Test that a buy signal is triggered when the momentum is positive."""
        self.mock_trading_env.get_momentum.return_value = 1.5  # Positive momentum
        signal = self.momentum_strategy.get_signal()
        self.assertEqual(signal, "buy")
        print(f"Buy signal: {signal}")

    def test_sell_signal(self):
        """Test that a sell signal is triggered when the momentum is negative."""
        self.mock_trading_env.get_momentum.return_value = -1.2  # Negative momentum
        signal = self.momentum_strategy.get_signal()
        self.assertEqual(signal, "sell")
        print(f"Sell signal: {signal}")

if __name__ == "__main__":
    unittest.main()
