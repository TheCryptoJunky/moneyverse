import unittest
from src.ai.reinforcement_learning_agent import RLTradingAgent

class MockEnvironment:
    # Mock environment for testing
    def reset(self):
        return [0.0]

    def step(self, action):
        return [0.0], 0.0, False, {}

class TestRLTradingAgent(unittest.TestCase):

    def setUp(self):
        self.env = MockEnvironment()
        self.agent = RLTradingAgent(environment=self.env, feature_inputs={})

    def test_initial_weights(self):
        expected_weights = {
            "lstm": 0.3,
            "sentiment": 0.2,
            "arima": 0.2,
            "technical": 0.2,
            "market_regime": 0.1
        }
        self.assertEqual(self.agent.current_weights, expected_weights)

    def test_adjust_weights_positive_performance(self):
        performance_metrics = {
            "lstm": 0.5,
            "sentiment": -0.3,
            "arima": 0.1,
            "technical": -0.2,
            "market_regime": 0.4
        }
        self.agent.adjust_weights(performance_metrics)
        expected_weights = {
            "lstm": 0.4,
            "sentiment": 0.1,
            "arima": 0.3,
            "technical": 0.1,
            "market_regime": 0.2
        }
        self.assertEqual(self.agent.current_weights, expected_weights)

    def test_adjust_weights_weight_limits(self):
        performance_metrics = {
            "lstm": 10,    # Simulate very high performance
            "sentiment": -10,  # Simulate very poor performance
        }
        self.agent.adjust_weights(performance_metrics)
        self.assertLessEqual(self.agent.current_weights["lstm"], 1.0)
        self.assertGreaterEqual(self.agent.current_weights["sentiment"], 0.0)

    def test_get_weighted_input(self):
        observations = {
            "lstm": 0.5,
            "sentiment": 0.3,
            "arima": 0.2,
            "technical": 0.4,
            "market_regime": 0.1
        }
        weighted_input = self.agent.get_weighted_input(observations)
        expected_weighted_input = (
            0.3 * 0.5 + 0.2 * 0.3 + 0.2 * 0.2 + 0.2 * 0.4 + 0.1 * 0.1
        )
        self.assertAlmostEqual(weighted_input, expected_weighted_input)

    def test_rebalance_phase(self):
        self.agent.rebalance_phase()
        expected_weights = {
            "lstm": 0.1,
            "sentiment": 0.1,
            "arima": 0.4,
            "technical": 0.1,
            "market_regime": 0.3
        }
        self.assertEqual(self.agent.current_weights, expected_weights)

if __name__ == '__main__':
    unittest.main()
