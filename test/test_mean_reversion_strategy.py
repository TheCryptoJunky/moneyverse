import unittest
from unittest.mock import Mock
from src.ai.mean_reversion_strategy import MeanReversionStrategy

class TestMeanReversionStrategy(unittest.TestCase):
    """Unit tests for the MeanReversionStrategy."""

    def setUp(self):
        """Set up the mean reversion strategy and mock the trading environment."""
        self.mock_trading_env = Mock()
        self.mean_reversion_strategy = MeanReversionStrategy(self.mock_trading_env)

    def test_buy_signal(self):
        """Test the buy signal when the price is below the moving average."""
        # Mock environment and moving average
        self.mock_trading_env.get_current_price.return_value = 95
        self.mock_trading_env.get_moving_average.return_value = 100

        # Check the buy signal
        signal = self.mean_reversion_strategy.get_signal()
        self.assertEqual(signal, "buy")
        print(f"Buy signal: {signal}")

    def test_sell_signal(self):
        """Test the sell signal when the price is above the moving average."""
        # Mock environment and moving average
        self.mock_trading_env.get_current_price.return_value = 105
        self.mock_trading_env.get_moving_average.return_value = 100

        # Check the sell signal
        signal = self.mean_reversion_strategy.get_signal()
        self.assertEqual(signal, "sell")
        print(f"Sell signal: {signal}")

    def test_execute_mean_reversion_trade(self):
        """Test that the strategy executes a buy or sell based on the signal."""
        # Mock environment and signal
        self.mock_trading_env.get_current_price.return_value = 95
        self.mean_reversion_strategy.get_signal = Mock(return_value="buy")

        # Execute trade
        self.mean_reversion_strategy.execute_trade()

        # Assert that a buy order was placed
        self.mock_trading_env.buy.assert_called_once()

if __name__ == "__main__":
    unittest.main()
