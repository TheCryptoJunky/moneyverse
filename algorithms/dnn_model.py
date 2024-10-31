import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from ..database.db_connection import DatabaseConnection

class DNNModel:
    """
    Deep Neural Network (DNN) for complex pattern recognition in time-series data.
    
    Attributes:
    - db (DatabaseConnection): Database connection for logging predictions.
    - model (Sequential): Compiled DNN model.
    - scaler (MinMaxScaler): Scaler to normalize data for input.
    """

    def __init__(self, input_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)

    def _build_model(self):
        """
        Builds and compiles the DNN model architecture with dropout layers for regularization.
        
        Returns:
        - Sequential: Compiled DNN model.
        """
        model = Sequential([
            Dense(128, input_shape=(self.input_shape,), activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1, activation="linear")  # Output layer for regression
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        self.logger.info("DNN model built and compiled.")
        return model

    def train(self, X_train, y_train, epochs=20, batch_size=32):
        """
        Trains the DNN model on the provided dataset.
        
        Args:
        - X_train (np.ndarray): Training input data.
        - y_train (np.ndarray): Training target data.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        self.logger.info("DNN model trained on data.")

    def predict(self, X_test) -> np.ndarray:
        """
        Makes predictions on the provided test data.
        
        Args:
        - X_test (np.ndarray): Test data for prediction.
        
        Returns:
        - np.ndarray: Predictions.
        """
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        self.logger.info("Generated predictions using DNN model.")
        return predictions

    def update_predictions(self, asset: str, predictions: np.ndarray):
        """
        Logs predictions into the database for a given asset.
        
        Args:
        - asset (str): Asset symbol.
        - predictions (np.ndarray): Array of predicted values.
        """
        self.db.log_forecast(asset, predictions.tolist())
        self.logger.info(f"Predictions for {asset} updated in the database.")
