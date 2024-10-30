import logging
from .reinforcement_learning_agent import ReinforcementAgent
from .dqn import DQNAgent
from .lstm_model import LSTMModel
from .pgm import PGMModel
from ..database.db_connection import DatabaseConnection

class AIModel:
    """
    Central AI model manager that handles dynamic selection and execution of different AI models.
    
    Attributes:
    - db (DatabaseConnection): Database for logging performance.
    - reinforcement_agent (ReinforcementAgent): Agent for RL-based strategy optimization.
    - dqn_agent (DQNAgent): Deep Q-Network agent for Q-learning.
    - lstm_model (LSTMModel): LSTM model for time-series forecasting.
    - pgm_model (PGMModel): Bayesian Network model for probabilistic reasoning.
    """

    def __init__(self, db: DatabaseConnection, state_size: int, action_size: int):
        self.db = db
        self.reinforcement_agent = ReinforcementAgent()
        self.dqn_agent = DQNAgent(state_size, action_size, db)
        self.lstm_model = LSTMModel(input_shape=(state_size, 1))
        self.pgm_model = PGMModel(db)
        self.logger = logging.getLogger(__name__)
        self.logger.info("AI Model Manager initialized with RL, DQN, LSTM, and PGM models.")

    def select_model(self, market_condition: str):
        """
        Selects the most suitable model based on the current market condition.
        
        Args:
        - market_condition (str): Current market condition ("volatile", "trending", "neutral").
        
        Returns:
        - Object: Selected model instance.
        """
        if market_condition == "volatile":
            model = self.pgm_model
            self.logger.info("Selected PGM model for volatile conditions.")
        elif market_condition == "trending":
            model = self.lstm_model
            self.logger.info("Selected LSTM model for trending conditions.")
        else:
            model = self.dqn_agent
            self.logger.info("Selected DQN model for neutral conditions.")
        return model

    def execute_model(self, model, data, target=None):
        """
        Executes the selected model on the provided data.

        Args:
        - model (Object): Selected model instance.
        - data (np.ndarray): Input data for the model.
        - target (np.ndarray, optional): Target data for supervised models like LSTM.
        
        Returns:
        - Object: Model output or prediction.
        """
        if isinstance(model, DQNAgent):
            action = model.act(data)
            self.logger.info(f"DQNAgent selected action: {action}")
            return action
        elif isinstance(model, LSTMModel):
            prediction = model.adapt_and_predict(data, target, data)
            self.logger.info(f"LSTMModel prediction: {prediction}")
            return prediction
        elif isinstance(model, PGMModel):
            market_trend = data.get("market_trend")
            volatility = data.get("volatility")
            prediction = model.predict_market_movement(market_trend, volatility)
            self.logger.info(f"PGMModel prediction: {prediction}")
            return prediction

    def log_model_performance(self, model_type: str, performance: dict):
        """
        Logs the model performance metrics to the database.
        
        Args:
        - model_type (str): Type of the model ("DQN", "LSTM", "PGM").
        - performance (dict): Performance metrics.
        """
        self.db.log_model_performance(model_type, performance)
        self.logger.info(f"Logged {model_type} performance: {performance}")

    def apply_best_model(self, market_condition: str, data, target=None):
        """
        Selects and executes the best model based on current market conditions, logs results.
        
        Args:
        - market_condition (str): Current market condition.
        - data (np.ndarray): Input data for the model.
        - target (np.ndarray, optional): Target data for supervised models like LSTM.
        
        Returns:
        - Object: Model output or prediction.
        """
        model = self.select_model(market_condition)
        output = self.execute_model(model, data, target)
        performance = {"model_type": model.__class__.__name__, "output": output}
        self.log_model_performance(model.__class__.__name__, performance)
        self.logger.info(f"Applied best model {model.__class__.__name__} based on {market_condition} market condition.")
        return output
