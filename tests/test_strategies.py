import unittest
from unittest.mock import Mock, patch
from src.strategies import (
    prioritize_greenlist,
    apply_dca_strategy,
    apply_greenlist_logic
)
from src.managers.wallet_manager import WalletManager

class TestStrategies(unittest.TestCase):

    @patch('src.strategies.execute_trade')
    def test_apply_dca_strategy(self, mock_execute_trade):
        mock_wallet_manager = Mock(spec=WalletManager)
        mock_wallet_manager.get_available_balance.return_value = 1000
        apply_dca_strategy('asset_address_example', mock_wallet_manager)
        trade_size = 1000 * 0.01  # 1% of balance
        mock_execute_trade.assert_called_with('asset_address_example', trade_size)

    @patch('src.strategies.execute_triangle_arbitrage')
    @patch('src.strategies.execute_cross_chain_arbitrage')
    @patch('src.strategies.apply_dca_strategy')
    @patch('src.strategies.insert_greenlist_log')
    def test_prioritize_greenlist(
        self,
        mock_insert_greenlist_log,
        mock_apply_dca_strategy,
        mock_execute_cross_chain_arbitrage,
        mock_execute_triangle_arbitrage
    ):
        mock_wallet_manager = Mock(spec=WalletManager)
        with patch('time.time', side_effect=[0, 10, 20, 30, 40, 50, 60]):
            with patch('time.sleep', return_value=None):
                prioritize_greenlist('asset_address_example', duration=1, wallet_manager=mock_wallet_manager)

        self.assertTrue(mock_apply_dca_strategy.called)
        self.assertTrue(mock_execute_triangle_arbitrage.called)
        self.assertTrue(mock_execute_cross_chain_arbitrage.called)
        self.assertTrue(mock_insert_greenlist_log.called)

    @patch('src.strategies.strategies.get_greenlist')
    @patch('src.strategies.strategies.prioritize_greenlist')
    def test_apply_greenlist_logic(self, mock_prioritize_greenlist, mock_get_greenlist):
        mock_wallet_manager = Mock(spec=WalletManager)
        mock_get_greenlist.return_value = [
            {'asset_address': 'asset1', 'focus_duration': 10},
            {'asset_address': 'asset2', 'focus_duration': 5}
        ]
        apply_greenlist_logic(mock_wallet_manager)
        self.assertEqual(mock_prioritize_greenlist.call_count, 2)
        mock_prioritize_greenlist.assert_any_call('asset1', 10, mock_wallet_manager)
        mock_prioritize_greenlist.assert_any_call('asset2', 5, mock_wallet_manager)

if __name__ == '__main__':
    unittest.main()
