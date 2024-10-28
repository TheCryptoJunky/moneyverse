# tests/test_arbitrage_strategy.py

import unittest
from unittest.mock import Mock
from src.ai.arbitrage_strategy import ArbitrageStrategy

class TestArbitrageStrategy(unittest.TestCase):
    """
    Unit tests for the ArbitrageStrategy class.
    """

    def setUp(self):
        """
        Set up mock APIs for DEX and centralized exchange.
        """
        self.mock_dex_api = Mock()
        self.mock_central_exchange_api = Mock()
        self.arbitrage_strategy = ArbitrageStrategy(self.mock_dex_api, self.mock_central_exchange_api)

    def test_detect_arbitrage_opportunity(self):
        """
        Test arbitrage detection when there is a price difference.
        """
        asset_id = "BTC"
        self.mock_dex_api.get_price.return_value = 50000
        self.mock_central_exchange_api.get_price.return_value = 51000

        # Call the method to detect arbitrage
        self.arbitrage_strategy.detect_arbitrage_opportunity(asset_id)

        # Verify the buy/sell methods were called correctly
        self.mock_dex_api.buy.assert_called_with(asset_id, 50000)
        self.mock_central_exchange_api.sell.assert_called_with(asset_id, 51000)

    def test_no_arbitrage_opportunity(self):
        """
        Test when there is no arbitrage opportunity due to insufficient price difference.
        """
        asset_id = "BTC"
        self.mock_dex_api.get_price.return_value = 50000
        self.mock_central_exchange_api.get_price.return_value = 50005

        # Call the method
        self.arbitrage_strategy.detect_arbitrage_opportunity(asset_id)

        # Assert no trade is executed
        self.mock_dex_api.buy.assert_not_called()
        self.mock_central_exchange_api.sell.assert_not_called()

if __name__ == "__main__":
    unittest.main()
