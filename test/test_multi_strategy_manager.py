import unittest
from src.ai.multi_strategy_manager import MultiStrategyManager
from unittest.mock import Mock

class TestMultiStrategyManager(unittest.TestCase):
    """Unit tests for the MultiStrategyManager."""

    def setUp(self):
        """Set up mock strategies and the multi-strategy manager."""
        self.mock_strategy1 = Mock()
        self.mock_strategy2 = Mock()
        self.strategies = [self.mock_strategy1, self.mock_strategy2]
        self.multi_strategy_manager = MultiStrategyManager(self.strategies)

    def test_execute_all_strategies(self):
        """Test if all strategies are executed properly."""
        self.multi_strategy_manager.execute_all()

        # Assert both strategies' execute methods are called
        self.mock_strategy1.execute.assert_called_once()
        self.mock_strategy2.execute.assert_called_once()

if __name__ == "__main__":
    unittest.main()
