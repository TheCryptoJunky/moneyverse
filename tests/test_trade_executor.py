import unittest
from unittest.mock import Mock
from src.trading.trade_executor import TradeExecutor

class TestTradeExecutor(unittest.TestCase):

    def setUp(self):
        self.mock_api = Mock()
        self.trade_executor = TradeExecutor(trading_api=self.mock_api)

    def test_execute_buy_trade(self):
        self.mock_api.buy.return_value = {'status': 'success', 'action': 'buy'}
        result = self.trade_executor.execute_trade('BTC', 'buy', 1.5)
        self.mock_api.buy.assert_called_with('BTC', 1.5)
        self.assertEqual(result['action'], 'buy')

    def test_execute_sell_trade(self):
        self.mock_api.sell.return_value = {'status': 'success', 'action': 'sell'}
        result = self.trade_executor.execute_trade('ETH', 'sell', 2.0)
        self.mock_api.sell.assert_called_with('ETH', 2.0)
        self.assertEqual(result['action'], 'sell')

    def test_execute_trade_invalid_action(self):
        with self.assertRaises(ValueError) as context:
            self.trade_executor.execute_trade('BTC', 'hold', 1.0)
        self.assertEqual(str(context.exception), "Unknown action: hold")

if __name__ == '__main__':
    unittest.main()
