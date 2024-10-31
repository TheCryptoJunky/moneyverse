import logging
from .reinforcement_learning_agent import ReinforcementLearningAgent
from .dnn_model import DNNModel
from .lstm_model import LSTMModel
from .pgm import PGMModel
from ..database.db_connection import DatabaseConnection

class AIModel:
    """
    Central AI model manager to dynamically manage and deploy different AI models.

    Attributes:
    - db (DatabaseConnection): Database for logging and tracking performance.
    - rl_agent (ReinforcementLearningAgent): Reinforcement learning agent.
    - dnn_model (DNNModel): Deep neural network for complex forecasting.
    - lstm_model (LSTMModel): LSTM model for time-series forecasting.
    - pgm_model (PGMModel): Bayesian network for probabilistic reasoning.
    """

    def __init__(self, db: DatabaseConnection, state_size: int, action_size: int):
        self.db = db
        self.rl_agent = ReinforcementLearningAgent(state_size, action_size, db)
        self.dnn_model = DNNModel(input_shape=(state_size,))
        self.lstm_model = LSTMModel(input_shape=(state_size, 1))
        self.pgm_model = PGMModel(db)
        self.logger = logging.getLogger(__name__)
        self.logger.info("AI Model manager initialized with RL, DNN, LSTM, and PGM models.")

    def select_model(self, market_condition: str):
        """
        Selects the appropriate model based on market conditions.

        Args:
        - market_condition (str): Current market condition (e.g., "volatile", "trending", "neutral").
        
        Returns:
        - Object: The selected model instance.
        """
        if market_condition == "volatile":
            model = self.pgm_model
            self.logger.info("Selected PGM model for volatile conditions.")
        elif market_condition == "trending":
            model = self.lstm_model
            self.logger.info("Selected LSTM model for trending conditions.")
        elif market_condition == "complex_patterns":
            model = self.dnn_model
            self.logger.info("Selected DNN model for complex pattern recognition.")
        else:
            model = self.rl_agent
            self.logger.info("Selected RL agent for neutral or learning conditions.")
        return model

    def execute_model(self, model, data, target=None):
        """
        Executes the selected model with the provided data.

        Args:
        - model (Object): Model to execute.
        - data (np.ndarray): Input data for the model.
        - target (np.ndarray, optional): Target data for supervised models.
        
        Returns:
        - Any: Output or prediction of the model.
        """
        if isinstance(model, ReinforcementLearningAgent):
            action = model.act(data)
            self.logger.info(f"RL Agent selected action: {action}")
            return action
        elif isinstance(model, DNNModel) or isinstance(model, LSTMModel):
            prediction = model.predict(data)
            self.logger.info(f"Prediction from {model.__class__.__name__}: {prediction}")
            return prediction
        elif isinstance(model, PGMModel):
            market_trend = data.get("market_trend")
            volatility = data.get("volatility")
            prediction = model.predict_market_movement(market_trend, volatility)
            self.logger.info(f"PGMModel prediction: {prediction}")
            return prediction

    def log_model_performance(self, model_type: str, performance: dict):
        """
        Logs performance metrics to the database.

        Args:
        - model_type (str): Type of model ("RL", "DNN", "LSTM", "PGM").
        - performance (dict): Performance metrics to log.
        """
        self.db.log_model_performance(model_type, performance)
        self.logger.info(f"Logged performance for {model_type}: {performance}")

    def apply_best_model(self, market_condition: str, data, target=None):
        """
        Applies the best model based on the market condition and logs performance.

        Args:
        - market_condition (str): Current market condition.
        - data (np.ndarray): Input data.
        - target (np.ndarray, optional): Target data for supervised models.

        Returns:
        - Any: Output or prediction from the applied model.
        """
        model = self.select_model(market_condition)
        output = self.execute_model(model, data, target)
        performance = {"model_type": model.__class__.__name__, "output": output}
        self.log_model_performance(model.__class__.__name__, performance)
        self.logger.info(f"Applied model {model.__class__.__name__} for {market_condition} conditions.")
        return output
