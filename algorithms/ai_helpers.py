import numpy as np
import logging
from decimal import Decimal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..database.db_connection import DatabaseConnection

class PredictionHelper:
    """
    Assists in prediction scaling, normalization, and aggregation across models.
    """

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes input data for model compatibility.

        Args:
        - data (np.ndarray): Raw input data to be normalized.
        
        Returns:
        - np.ndarray: Normalized data.
        """
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        logging.info("Data normalized for model input.")
        return normalized

    def aggregate_predictions(self, predictions: list, weights: list = None) -> float:
        """
        Aggregates predictions from multiple models using weighted averaging.

        Args:
        - predictions (list): List of model predictions.
        - weights (list): Optional weights for averaging.
        
        Returns:
        - float: Weighted average prediction.
        """
        if weights is None:
            weights = [1 / len(predictions)] * len(predictions)
        combined_prediction = sum(p * w for p, w in zip(predictions, weights))
        logging.info(f"Aggregated predictions with weights {weights}: {combined_prediction}")
        return combined_prediction

class ReinforcementLearningHelper:
    """
    Provides tools for exploration strategies, reward processing, and epsilon decay.
    """

    def epsilon_decay(self, epsilon: float, decay_rate: float, min_epsilon: float) -> float:
        """
        Adjusts epsilon to balance exploration and exploitation over time.

        Args:
        - epsilon (float): Current epsilon value.
        - decay_rate (float): Epsilon decay rate.
        - min_epsilon (float): Minimum epsilon value.
        
        Returns:
        - float: Updated epsilon.
        """
        new_epsilon = max(min_epsilon, epsilon * decay_rate)
        logging.info(f"Epsilon decayed to {new_epsilon}")
        return new_epsilon

    def reward_normalization(self, reward: float, scaling_factor: float = 0.01) -> float:
        """
        Normalizes rewards to improve training stability.

        Args:
        - reward (float): Raw reward value.
        - scaling_factor (float): Factor for scaling rewards.
        
        Returns:
        - float: Scaled reward.
        """
        normalized_reward = reward * scaling_factor
        logging.info(f"Reward normalized to {normalized_reward}")
        return normalized_reward

class TokenSafetyHelper:
    """
    Ensures safe token transactions by managing gas fees, volatility, and sentiment.
    """

    def adjust_gas_limit(self, base_gas: int, volatility: float) -> int:
        """
        Dynamically adjusts the gas limit based on market volatility.

        Args:
        - base_gas (int): Base gas limit.
        - volatility (float): Current market volatility.
        
        Returns:
        - int: Adjusted gas limit.
        """
        adjusted_gas = int(base_gas * (1 + volatility))
        logging.info(f"Gas limit adjusted to {adjusted_gas} based on volatility {volatility}")
        return adjusted_gas

    def sentiment_analysis(self, text_data: list) -> float:
        """
        Analyzes sentiment in text data to gauge overall market mood.

        Args:
        - text_data (list): List of text inputs.
        
        Returns:
        - float: Average sentiment score.
        """
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(text)["compound"] for text in text_data]
        average_score = sum(scores) / len(scores) if scores else 0.0
        logging.info(f"Calculated sentiment score: {average_score}")
        return average_score
