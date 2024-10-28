# tests/test_trade_execution_integration.py

import unittest
from unittest.mock import Mock
from src.trading.trade_executor import TradeExecutor

class TestTradeExecutionIntegration(unittest.TestCase):
    """
    Integration tests for trade execution.
    """

    def setUp(self):
        """Set up a mock trading API."""
        self.mock_trading_api = Mock()
        self.executor = TradeExecutor(self.mock_trading_api)

    def test_execute_buy_trade(self):
        """Test a buy trade is executed correctly."""
        self.executor.execute_trade("BTC", "buy", 1)
        self.mock_trading_api.buy.assert_called_with("BTC", 1)

    def test_execute_sell_trade(self):
        """Test a sell trade is executed correctly."""
        self.executor.execute_trade("BTC", "sell", 1)
        self.mock_trading_api.sell.assert_called_with("BTC", 1)

if __name__ == "__main__":
    unittest.main()
