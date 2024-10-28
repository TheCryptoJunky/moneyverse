import unittest
from unittest.mock import Mock
from src.ai.multi_agent_manager import MultiAgentManager

class TestMultiAgentIntegration(unittest.TestCase):
    """Test suite for MultiAgentManager integration."""

    def setUp(self):
        """Set up mocks for strategies and agent configurations."""
        # Mock the agent configuration
        self.mock_agent_config = [{'market_data': 'mock_market_data'}]

        # Mock the strategies for the multi-agent manager
        self.mock_strategies = Mock()

        # Initialize MultiAgentManager with mock configuration and strategies
        self.manager = MultiAgentManager(self.mock_agent_config, self.mock_strategies)

    def test_execute_all(self):
        """Test that all strategies execute correctly."""
        self.manager.coordinate_agents()  # Simulate coordination

    def test_get_signals(self):
        """Test that the manager collects signals from all strategies."""
        signals = self.manager.get_signals()  # Get signals from agents
        self.assertIsNotNone(signals)

if __name__ == "__main__":
    unittest.main()
