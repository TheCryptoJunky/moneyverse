import logging
import numpy as np
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, State
from typing import Dict
from ..database.db_connection import DatabaseConnection

class PGMModel:
    """
    Probabilistic Graphical Model (PGM) for evaluating market trends and conditions.
    
    Attributes:
    - db (DatabaseConnection): Database for logging performance.
    - model (BayesianNetwork): Bayesian network for probabilistic reasoning.
    """

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.model = self._build_model()
        self.logger = logging.getLogger(__name__)
        self.logger.info("PGM Model initialized with Bayesian Network.")

    def _build_model(self):
        """
        Builds a Bayesian Network to model relationships between observed market variables.

        Returns:
        - BayesianNetwork: Configured Bayesian Network.
        """
        volatility = DiscreteDistribution({'low': 0.5, 'high': 0.5})
        trend = DiscreteDistribution({'bullish': 0.5, 'bearish': 0.5})

        market_condition = ConditionalProbabilityTable(
            [
                ['low', 'bullish', 'stable', 0.6],
                ['low', 'bullish', 'volatile', 0.4],
                ['low', 'bearish', 'stable', 0.7],
                ['low', 'bearish', 'volatile', 0.3],
                ['high', 'bullish', 'stable', 0.3],
                ['high', 'bullish', 'volatile', 0.7],
                ['high', 'bearish', 'stable', 0.4],
                ['high', 'bearish', 'volatile', 0.6]
            ],
            [volatility, trend]
        )

        s1 = State(volatility, name="volatility")
        s2 = State(trend, name="trend")
        s3 = State(market_condition, name="market_condition")

        model = BayesianNetwork("Market Trend Model")
        model.add_states(s1, s2, s3)
        model.add_edge(s1, s3)
        model.add_edge(s2, s3)
        model.bake()
        
        self.logger.info("Bayesian Network built for market condition analysis.")
        return model

    def predict_market_movement(self, volatility: str, trend: str) -> Dict[str, float]:
        """
        Predicts market movement given volatility and trend conditions.

        Args:
        - volatility (str): Observed volatility ("low" or "high").
        - trend (str): Observed trend ("bullish" or "bearish").

        Returns:
        - dict: Predicted probabilities for market conditions.
        """
        try:
            beliefs = self.model.predict_proba({'volatility': volatility, 'trend': trend})
            market_probs = beliefs[2].parameters[0]
            self.logger.info(f"Predicted market condition probabilities: {market_probs}")
            return market_probs
        except Exception as e:
            self.logger.error(f"Error in market movement prediction: {e}")
            return {}

    def log_prediction(self, conditions: Dict[str, str], prediction: Dict[str, float]):
        """
        Logs the prediction results for market condition analysis.

        Args:
        - conditions (dict): Observed conditions (e.g., volatility and trend).
        - prediction (dict): Prediction result with probabilities.
        """
        self.db.log_pgm_prediction(conditions, prediction)
        self.logger.info(f"Prediction logged with conditions {conditions} and results {prediction}.")
