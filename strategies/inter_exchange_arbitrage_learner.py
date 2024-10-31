import numpy as np
from typing import Dict, Any
from database.db_connection import DatabaseConnection
from self_learning.self_learning_engine import SelfLearningEngine
from strategies.inter_exchange_arbitrage import InterExchangeArbitrageBot

class InterExchangeArbitrageLearner:
    """
    Enhances InterExchangeArbitrageBot with self-learning capabilities for optimizing
    arbitrage decisions using reinforcement learning.
    """

    def __init__(self, db: DatabaseConnection):
        """
        Initialize the learner with database connection and AI components.

        Args:
            db (DatabaseConnection): Database connection for storing learning data
        """
        # Initialize self-learning engine
        self.learning_engine = SelfLearningEngine(
            db=db,
            state_size=64,  # Size of our market state representation
            action_size=3,   # Actions: 0=skip, 1=execute, 2=execute with flash loan
            buffer_size=10000
        )
        
        # Register strategy with custom reward function and state adapter
        self.learning_engine.register_strategy(
            strategy_name="inter_exchange_arbitrage",
            reward_function=self._calculate_arbitrage_reward,
            state_adapter=self._adapt_market_state
        )

    def _adapt_market_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        Converts raw market state into a normalized vector for the learning engine.

        Args:
            raw_state (Dict[str, Any]): Raw market state including prices, volumes, etc.

        Returns:
            np.ndarray: Normalized state vector
        """
        # Extract relevant features
        price_diff = raw_state.get("price_difference", 0)
        volume = raw_state.get("volume", 0)
        volatility = raw_state.get("volatility", 0)
        liquidity = raw_state.get("liquidity", 0)
        gas_price = raw_state.get("gas_price", 0)
        
        # Create feature vector
        features = np.array([
            price_diff,
            volume,
            volatility,
            liquidity,
            gas_price,
            # Add more features as needed
        ])
        
        # Normalize features
        normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
        return normalized

    def _calculate_arbitrage_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculates the reward for an arbitrage action based on profitability and risk.

        Args:
            state (Dict[str, Any]): Current market state

        Returns:
            float: Calculated reward
        """
        # Extract metrics
        profit = state.get("profit", 0)
        gas_cost = state.get("gas_cost", 0)
        slippage = state.get("slippage", 0)
        execution_time = state.get("execution_time", 1)
        
        # Calculate base reward from profit
        base_reward = profit - gas_cost
        
        # Apply penalties for risk factors
        slippage_penalty = -abs(slippage) * 0.1
        time_penalty = -np.log(execution_time) * 0.05
        
        # Combine rewards and penalties
        total_reward = base_reward + slippage_penalty + time_penalty
        return float(total_reward)

    async def enhance_arbitrage_bot(self, bot: InterExchangeArbitrageBot):
        """
        Enhances the arbitrage bot with learning capabilities.

        Args:
            bot (InterExchangeArbitrageBot): The bot to enhance
        """
        original_execute = bot.execute_inter_exchange_trade
        
        async def enhanced_execute_trade(opportunity: dict):
            """
            Enhanced trade execution with learning-based decision making.

            Args:
                opportunity (dict): The arbitrage opportunity
            """
            # Prepare market state
            current_state = {
                "price_difference": opportunity.get("sell_price", 0) - opportunity.get("buy_price", 0),
                "volume": opportunity.get("amount", 0),
                "volatility": opportunity.get("volatility", 0),
                "liquidity": opportunity.get("liquidity", 0),
                "gas_price": opportunity.get("gas_price", 0),
                # Add more state information as needed
            }
            
            # Get action from learning engine
            action, confidence = self.learning_engine.select_action(
                current_state,
                "inter_exchange_arbitrage"
            )
            
            # Execute based on learned action
            if action == 0:  # Skip
                bot.logger.info("Learning engine decided to skip opportunity")
                return
            
            # Execute trade
            success = await original_execute(opportunity)
            
            # Prepare next state and calculate metrics
            next_state = {
                **current_state,
                "profit": opportunity.get("expected_profit", 0) if success else 0,
                "gas_cost": opportunity.get("gas_cost", 0),
                "slippage": opportunity.get("actual_slippage", 0),
                "execution_time": opportunity.get("execution_time", 1),
            }
            
            # Update learning engine
            self.learning_engine.optimize_strategy(
                "inter_exchange_arbitrage",
                current_state,
                {
                    "market_data": current_state,
                    "action_result": success,
                    "confidence": confidence
                }
            )
            
            # Log learning progress
            bot.logger.info(
                f"Learning update - Action: {action}, Confidence: {confidence:.2f}, "
                f"Success: {success}"
            )
        
        # Replace original execution method with enhanced version
        bot.execute_inter_exchange_trade = enhanced_execute_trade
        bot.logger.info("Enhanced arbitrage bot with learning capabilities")

    async def train_on_historical_data(self, historical_data: list):
        """
        Trains the learning engine on historical arbitrage data.

        Args:
            historical_data (list): List of historical arbitrage opportunities and results
        """
        for data_point in historical_data:
            # Prepare state from historical data
            state = {
                "price_difference": data_point.get("price_diff", 0),
                "volume": data_point.get("volume", 0),
                "volatility": data_point.get("volatility", 0),
                "liquidity": data_point.get("liquidity", 0),
                "gas_price": data_point.get("gas_price", 0),
            }
            
            # Get actual action taken historically
            action = 1 if data_point.get("executed", False) else 0
            
            # Calculate historical reward
            reward = self._calculate_arbitrage_reward({
                "profit": data_point.get("profit", 0),
                "gas_cost": data_point.get("gas_cost", 0),
                "slippage": data_point.get("slippage", 0),
                "execution_time": data_point.get("execution_time", 1),
            })
            
            # Store experience
            self.learning_engine.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=state,  # Use same state as next state for historical data
                done=True
            )
        
        # Train models on collected experiences
        self.learning_engine.train_models(batch_size=32)
