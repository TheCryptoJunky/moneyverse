import unittest
import numpy as np
from unittest.mock import Mock, patch
from database.db_connection import DatabaseConnection
from self_learning.self_learning_engine import SelfLearningEngine
from algorithms.environment import TradingEnv

class TestSelfLearningEngine(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case"""
        # Mock database connection
        self.db = Mock(spec=DatabaseConnection)
        
        # Initialize self-learning engine with test parameters
        self.engine = SelfLearningEngine(
            db=self.db,
            learning_rate=0.01,
            discount_factor=0.95,
            state_size=32,
            action_size=4,
            buffer_size=1000
        )
        
        # Mock market data for testing
        self.test_market_data = {
            'prices': np.random.random(100),
            'volumes': np.random.random(100),
            'timestamps': list(range(100))
        }
        
        # Define test strategy
        def test_reward_func(state):
            return float(np.mean(state.get('prices', [0])))
        
        def test_state_adapter(raw_state):
            return np.array(raw_state.get('prices', [0]))
        
        # Register test strategy
        self.engine.register_strategy(
            'test_strategy',
            reward_function=test_reward_func,
            state_adapter=test_state_adapter
        )

    def test_initialization(self):
        """Test proper initialization of components"""
        self.assertIsNotNone(self.engine.replay_buffer)
        self.assertIsNotNone(self.engine.actor_critic)
        self.assertIsNotNone(self.engine.marl_agent)
        self.assertIsNotNone(self.engine.sentiment_analyzer)

    def test_trading_env_initialization(self):
        """Test trading environment initialization"""
        self.engine.initialize_trading_env(self.test_market_data)
        self.assertIsInstance(self.engine.trading_env, TradingEnv)

    def test_strategy_registration(self):
        """Test strategy registration functionality"""
        self.assertIn('test_strategy', self.engine.strategies)
        self.assertTrue(callable(self.engine.strategies['test_strategy']['reward_func']))
        self.assertTrue(callable(self.engine.strategies['test_strategy']['state_adapter']))

    def test_state_preprocessing(self):
        """Test state preprocessing with registered adapter"""
        test_state = {'prices': [1.0, 2.0, 3.0]}
        processed_state = self.engine.preprocess_state(test_state, 'test_strategy')
        self.assertIsInstance(processed_state, np.ndarray)
        self.assertEqual(len(processed_state), 3)

    def test_reward_calculation(self):
        """Test reward calculation with multiple components"""
        test_state = {'prices': [1.0, 2.0, 3.0]}
        test_next_state = {'prices': [1.5, 2.5, 3.5]}
        
        # Initialize trading env for reward calculation
        self.engine.initialize_trading_env(self.test_market_data)
        
        reward = self.engine.calculate_reward(
            'test_strategy',
            test_state,
            action=0,
            next_state=test_next_state
        )
        
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, 0.0)

    def test_experience_storage(self):
        """Test experience storage in replay buffer"""
        test_state = {'prices': [1.0, 2.0, 3.0]}
        test_next_state = {'prices': [1.5, 2.5, 3.5]}
        
        self.engine.store_experience(
            state=test_state,
            action=0,
            reward=1.0,
            next_state=test_next_state,
            done=False
        )
        
        self.assertEqual(len(self.engine.replay_buffer), 1)

    def test_action_selection(self):
        """Test action selection with exploration"""
        test_state = {'prices': [1.0, 2.0, 3.0]}
        
        # Test exploratory action
        action, confidence = self.engine.select_action(
            test_state,
            'test_strategy',
            epsilon=1.0
        )
        self.assertIsInstance(action, int)
        self.assertIsInstance(confidence, float)
        
        # Test exploitative action
        action, confidence = self.engine.select_action(
            test_state,
            'test_strategy',
            epsilon=0.0
        )
        self.assertIsInstance(action, int)
        self.assertIsInstance(confidence, float)

    def test_strategy_optimization(self):
        """Test complete strategy optimization cycle"""
        test_state = {'prices': [1.0, 2.0, 3.0]}
        
        # Mock database execution
        self.db.execute = Mock()
        
        self.engine.optimize_strategy(
            'test_strategy',
            test_state,
            self.test_market_data
        )
        
        # Verify database logging
        self.db.execute.assert_called_once()

    def test_model_training(self):
        """Test model training with batch"""
        # Fill replay buffer with some experiences
        for _ in range(50):
            test_state = {'prices': np.random.random(3)}
            test_next_state = {'prices': np.random.random(3)}
            self.engine.store_experience(
                state=test_state,
                action=0,
                reward=1.0,
                next_state=test_next_state,
                done=False
            )
        
        # Train models
        self.engine.train_models(batch_size=32)
        
        # Verify replay buffer was used
        self.assertGreaterEqual(len(self.engine.replay_buffer), 32)

if __name__ == '__main__':
    unittest.main()
