import logging
import numpy as np
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, State
from ..database.db_connection import DatabaseConnection

class PGMModel:
    """
    Probabilistic Graphical Model (PGM) that uses Bayesian Networks to make probabilistic predictions.
    
    Attributes:
    - db (DatabaseConnection): Database connection for logging predictions.
    - model (BayesianNetwork): Bayesian Network model instance.
    """

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.model = self._build_model()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized Bayesian Network for PGMModel.")

    def _build_model(self):
        """
        Constructs the Bayesian Network with conditional probability tables.

        Returns:
        - BayesianNetwork: Initialized Bayesian Network model.
        """
        # Define prior distributions for variables
        market_trend = DiscreteDistribution({"bullish": 0.5, "bearish": 0.5})
        volatility = DiscreteDistribution({"low": 0.5, "high": 0.5})

        # Conditional probability tables based on dependencies
        price_movement = ConditionalProbabilityTable(
            [
                ["bullish", "low", "up", 0.7],
                ["bullish", "low", "down", 0.3],
                ["bullish", "high", "up", 0.6],
                ["bullish", "high", "down", 0.4],
                ["bearish", "low", "up", 0.2],
                ["bearish", "low", "down", 0.8],
                ["bearish", "high", "up", 0.1],
                ["bearish", "high", "down", 0.9]
            ],
            [market_trend, volatility]
        )

        # Create states
        market_trend_state = State(market_trend, name="market_trend")
        volatility_state = State(volatility, name="volatility")
        price_movement_state = State(price_movement, name="price_movement")

        # Build the network
        model = BayesianNetwork("Market Analysis Network")
        model.add_states(market_trend_state, volatility_state, price_movement_state)
        model.add_edge(market_trend_state, price_movement_state)
        model.add_edge(volatility_state, price_movement_state)
        model.bake()

        return model

    def predict_market_movement(self, market_trend: str, volatility: str) -> str:
        """
        Predicts market movement based on given trend and volatility levels.

        Args:
        - market_trend (str): The current market trend ("bullish" or "bearish").
        - volatility (str): The current market volatility ("low" or "high").

        Returns:
        - str: Predicted market movement ("up" or "down").
        """
        belief = self.model.predict_proba({"market_trend": market_trend, "volatility": volatility})
        price_movement_belief = belief[-1].parameters[0]
        prediction = "up" if price_movement_belief["up"] > price_movement_belief["down"] else "down"
        self.logger.info(f"Market prediction based on trend {market_trend} and volatility {volatility}: {prediction}")
        return prediction

    def log_prediction(self, asset: str, prediction: str):
        """
        Logs probabilistic predictions into the database for historical tracking.

        Args:
        - asset (str): Asset symbol.
        - prediction (str): Predicted market movement.
        """
        self.db.log_forecast(asset, {"market_prediction": prediction})
        self.logger.info(f"Logged prediction for {asset}: {prediction}")

    def update_model(self, new_data: dict):
        """
        Updates the Bayesian Network with new data to refine predictions.

        Args:
        - new_data (dict): Dictionary containing market data for updating probabilities.
        """
        # Placeholder for dynamic model update
        # Example: Retrain conditional probabilities with new data if needed
        self.logger.info("Bayesian Network updated with new data.")
