import logging
from typing import Dict, Callable, List, Tuple, Any
import numpy as np
from algorithms.actor_critic import ActorCriticAgent
from algorithms.marl import MARLAgent
from algorithms.environment import TradingEnv
from algorithms.replay_buffer import ReplayBuffer
from algorithms.sentiment_analysis import SentimentAnalyzer
from database.db_connection import DatabaseConnection

class SelfLearningEngine:
    """
    Enhanced self-learning engine that integrates multiple AI models and strategies
    for sophisticated trading optimization.

    Attributes:
        strategies (dict): Registered strategies with their associated reward functions
        learning_rate (float): Controls the weight of updates in learning algorithms
        discount_factor (float): Discount factor for future rewards
        replay_buffer (ReplayBuffer): Storage for experience replay
        actor_critic (ActorCriticAgent): Main policy learning agent
        marl_agent (MARLAgent): Multi-agent system for collaborative learning
        sentiment_analyzer (SentimentAnalyzer): Market sentiment analysis
        trading_env (TradingEnv): Standardized trading environment
        db (DatabaseConnection): Database connection for persistent storage
    """

    def __init__(self, db: DatabaseConnection, learning_rate=0.1, discount_factor=0.9,
                 state_size=64, action_size=8, buffer_size=10000):
        self.db = db
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.strategies = {}
        
        # Initialize components
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.actor_critic = ActorCriticAgent(state_size, action_size)
        self.marl_agent = MARLAgent(num_agents=3, state_size=state_size, 
                                  action_size=action_size, db=db)
        self.sentiment_analyzer = SentimentAnalyzer(db)
        self.trading_env = None  # Initialized when market data is provided
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced SelfLearningEngine initialized with AI components")

    def initialize_trading_env(self, market_data: Dict[str, Any]):
        """
        Initializes or updates the trading environment with new market data.

        Args:
            market_data (Dict[str, Any]): Current market state and historical data
        """
        self.trading_env = TradingEnv(market_data)
        self.logger.info("Trading environment initialized with current market data")

    def register_strategy(self, strategy_name: str, reward_function: Callable,
                         state_adapter: Callable = None):
        """
        Registers a strategy with its reward function and optional state adapter.

        Args:
            strategy_name (str): Name of the strategy
            reward_function (callable): Function that calculates reward
            state_adapter (callable, optional): Function to convert raw state to normalized form
        """
        self.strategies[strategy_name] = {
            'reward_func': reward_function,
            'state_adapter': state_adapter or (lambda x: x)
        }
        self.logger.info(f"Registered strategy {strategy_name} with custom adapters")

    def preprocess_state(self, raw_state: Dict[str, Any], strategy_name: str) -> np.ndarray:
        """
        Preprocesses the raw state using strategy-specific adapter.

        Args:
            raw_state (Dict[str, Any]): Raw state information
            strategy_name (str): Name of the strategy for specific adaptation

        Returns:
            np.ndarray: Processed state vector
        """
        adapter = self.strategies[strategy_name]['state_adapter']
        return adapter(raw_state)

    def calculate_reward(self, strategy_name: str, state: Dict[str, Any],
                        action: int, next_state: Dict[str, Any]) -> float:
        """
        Calculates comprehensive reward incorporating multiple factors.

        Args:
            strategy_name (str): Name of the strategy
            state (Dict[str, Any]): Current state
            action (int): Taken action
            next_state (Dict[str, Any]): Resulting state

        Returns:
            float: Calculated reward
        """
        if strategy_name not in self.strategies:
            self.logger.warning(f"No reward function found for strategy {strategy_name}")
            return 0.0

        # Calculate base reward from strategy
        base_reward = self.strategies[strategy_name]['reward_func'](state)
        
        # Incorporate sentiment analysis
        sentiment_score = self.sentiment_analyzer.analyze_sentiment(
            [str(state), str(next_state)]
        )
        
        # Calculate environment reward
        env_reward = self.trading_env._calculate_reward(action)
        
        # Combine rewards with weights
        total_reward = (0.5 * base_reward + 
                       0.3 * sentiment_score +
                       0.2 * env_reward)
        
        self.logger.debug(f"Reward breakdown - Base: {base_reward:.2f}, "
                         f"Sentiment: {sentiment_score:.2f}, Env: {env_reward:.2f}")
        return total_reward

    def store_experience(self, state: Dict[str, Any], action: int,
                        reward: float, next_state: Dict[str, Any], done: bool):
        """
        Stores experience in replay buffer and updates all learning components.

        Args:
            state (Dict[str, Any]): Current state
            action (int): Taken action
            reward (float): Received reward
            next_state (Dict[str, Any]): Next state
            done (bool): Whether episode is complete
        """
        # Store in replay buffer
        self.replay_buffer.store((state, action, reward, next_state, done))
        
        # Update actor-critic
        self.actor_critic.store_experience(state, action, reward, next_state, done)
        
        # Update MARL system
        self.marl_agent.store_experience([state], [action], [reward],
                                       [next_state], [done])

    def train_models(self, batch_size: int = 32):
        """
        Trains all AI models using stored experiences.

        Args:
            batch_size (int): Size of training batch
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Train actor-critic
        self.actor_critic.train(batch_size)
        
        # Train MARL agents
        self.marl_agent.train_agents(batch_size)
        
        self.logger.info(f"Completed training iteration with batch size {batch_size}")

    def select_action(self, state: Dict[str, Any], strategy_name: str,
                     epsilon: float = 0.1) -> Tuple[int, float]:
        """
        Selects action using ensemble of models.

        Args:
            state (Dict[str, Any]): Current state
            strategy_name (str): Active strategy name
            epsilon (float): Exploration rate

        Returns:
            Tuple[int, float]: Selected action and its confidence score
        """
        if np.random.rand() < epsilon:
            action = np.random.randint(self.actor_critic.action_size)
            confidence = epsilon
            self.logger.debug(f"Exploratory action selected: {action}")
        else:
            # Get predictions from different models
            ac_action = self.actor_critic.act(state)
            marl_action = self.marl_agent.select_joint_action([state])[0]
            
            # Ensemble decision
            actions = [ac_action, marl_action]
            confidence = sum(actions) / len(actions)
            action = int(round(confidence))
            
            self.logger.debug(f"Ensemble action selected: {action} "
                            f"with confidence {confidence:.2f}")
        
        return action, confidence

    def optimize_strategy(self, strategy_name: str, current_state: Dict[str, Any],
                         market_data: Dict[str, Any]):
        """
        Runs comprehensive strategy optimization using all AI components.

        Args:
            strategy_name (str): Name of the strategy being optimized
            current_state (Dict[str, Any]): Current market state
            market_data (Dict[str, Any]): Additional market data
        """
        # Update trading environment
        self.initialize_trading_env(market_data)
        
        # Preprocess state
        processed_state = self.preprocess_state(current_state, strategy_name)
        
        # Select action
        action, confidence = self.select_action(processed_state, strategy_name)
        
        # Execute action in environment
        next_state, reward, done, _ = self.trading_env.step(action)
        
        # Calculate comprehensive reward
        total_reward = self.calculate_reward(strategy_name, current_state,
                                           action, next_state)
        
        # Store experience
        self.store_experience(processed_state, action, total_reward,
                            next_state, done)
        
        # Train models
        self.train_models()
        
        # Log optimization step
        self.logger.info(f"Optimized strategy {strategy_name} - "
                        f"Action: {action}, Confidence: {confidence:.2f}, "
                        f"Reward: {total_reward:.2f}")

        # Store optimization results in database
        self.db.execute(
            """
            INSERT INTO strategy_optimization_log 
            (strategy_name, action, confidence, reward, timestamp)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (strategy_name, action, confidence, total_reward)
        )
