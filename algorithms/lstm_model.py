import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from ..database.db_connection import DatabaseConnection

class LSTMModel:
    """
    Long Short-Term Memory (LSTM) model for time-series forecasting in trading.
    
    Attributes:
    - db (DatabaseConnection): Database connection for logging predictions.
    - model (Sequential): Compiled LSTM model.
    - scaler (MinMaxScaler): Scaler to normalize data for LSTM input.
    """

    def __init__(self, input_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)

    def _build_model(self):
        """
        Builds and compiles the LSTM model architecture.
        
        Returns:
        - keras.Sequential: Compiled LSTM model.
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation="relu"),
            Dense(1)  # Forecast single future value
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        self.logger.info("LSTM model built and compiled.")
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Trains the LSTM model on the provided data.
        
        Args:
        - X_train (np.ndarray): Training input data.
        - y_train (np.ndarray): Training output data.
        - epochs (int): Number of epochs for training.
        - batch_size (int): Batch size for training.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        self.logger.info("LSTM model trained on new data.")

    def predict(self, X_test) -> np.ndarray:
        """
        Makes predictions on the provided test data.
        
        Args:
        - X_test (np.ndarray): Input data for prediction.
        
        Returns:
        - np.ndarray: Scaled predictions.
        """
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        self.logger.info("Generated predictions using LSTM model.")
        return predictions

    def adapt_and_predict(self, X_train, y_train, X_test):
        """
        Adapts the model with recent training data and generates predictions.
        
        Args:
        - X_train (np.ndarray): Recent data for model adaptation.
        - y_train (np.ndarray): Target values for recent data.
        - X_test (np.ndarray): Test data for prediction.

        Returns:
        - np.ndarray: Scaled predictions after adaptation.
        """
        self.train(X_train, y_train, epochs=2)  # Quick adaptation with fewer epochs
        predictions = self.predict(X_test)
        self.logger.info("Model adapted to recent data and generated predictions.")
        return predictions

    def update_predictions(self, asset: str, predictions: np.ndarray):
        """
        Logs predictions into the database for the specified asset.
        
        Args:
        - asset (str): Asset symbol.
        - predictions (np.ndarray): Array of predicted values.
        """
        self.db.log_forecast(asset, predictions.tolist())
        self.logger.info(f"Predictions for {asset} updated in the database.")
