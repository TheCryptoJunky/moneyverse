import numpy as np
import logging
from decimal import Decimal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..database.db_connection import DatabaseConnection

class PredictionHelper:
    """
    Assists in prediction scaling, normalization, and multi-source data integration.
    """

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes input data for models requiring scaled inputs.
        
        Args:
        - data (np.ndarray): Input data to normalize.
        
        Returns:
        - np.ndarray: Normalized data.
        """
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        logging.info("Data normalized for model input.")
        return normalized

    def combine_predictions(self, predictions: list, weights: list = None) -> float:
        """
        Combines multiple predictions using specified weights.
        
        Args:
        - predictions (list): List of model predictions.
        - weights (list): List of weights for weighted averaging.
        
        Returns:
        - float: Weighted average of predictions.
        """
        if weights is None:
            weights = [1 / len(predictions)] * len(predictions)
        combined_prediction = sum(p * w for p, w in zip(predictions, weights))
        logging.info(f"Combined predictions with weights {weights}: {combined_prediction}")
        return combined_prediction

class ReinforcementLearningHelper:
    """
    Provides functions for reinforcement learning exploration and reward scaling.
    """

    def epsilon_decay(self, epsilon: float, decay_rate: float, min_epsilon: float) -> float:
        """
        Decays epsilon to balance exploration and exploitation.
        
        Args:
        - epsilon (float): Current exploration rate.
        - decay_rate (float): Rate of decay.
        - min_epsilon (float): Minimum allowable epsilon.
        
        Returns:
        - float: Updated epsilon.
        """
        new_epsilon = max(min_epsilon, epsilon * decay_rate)
        logging.info(f"Epsilon decayed to {new_epsilon}")
        return new_epsilon

    def reward_scaling(self, reward: float, scaling_factor: float = 0.01) -> float:
        """
        Scales the reward to ensure stability during training.
        
        Args:
        - reward (float): Original reward.
        - scaling_factor (float): Scaling factor.
        
        Returns:
        - float: Scaled reward.
        """
        scaled_reward = reward * scaling_factor
        logging.info(f"Reward scaled to {scaled_reward}")
        return scaled_reward

class TokenSafetyHelper:
    """
    Monitors and ensures the safety of token transactions with gas and volatility checks.
    """

    def calculate_gas_limit(self, base_gas: int, volatility: float) -> int:
        """
        Adjusts gas limit based on market volatility.
        
        Args:
        - base_gas (int): Base gas limit.
        - volatility (float): Current market volatility level.
        
        Returns:
        - int: Adjusted gas limit.
        """
        gas_limit = int(base_gas * (1 + volatility))
        logging.info(f"Gas limit adjusted to {gas_limit} based on volatility {volatility}")
        return gas_limit

    def sentiment_analysis(self, text_data: list) -> float:
        """
        Analyzes sentiment in text data to gauge market sentiment.
        
        Args:
        - text_data (list): List of text strings.
        
        Returns:
        - float: Average sentiment score.
        """
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(text)["compound"] for text in text_data]
        average_score = sum(scores) / len(scores) if scores else 0.0
        logging.info(f"Calculated average sentiment score: {average_score}")
        return average_score
