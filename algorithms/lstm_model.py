import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    """
    Long Short-Term Memory (LSTM) model for time-series forecasting of market data.

    Attributes:
    - input_shape (tuple): Shape of the input data.
    - scaler (MinMaxScaler): Scaler for input data normalization.
    - model (Sequential): Compiled LSTM model.
    - logger (Logger): Logger for tracking model operations.
    """

    def __init__(self, input_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        self.logger = logging.getLogger(__name__)
        self.logger.info("LSTM model initialized.")

    def _build_model(self):
        """
        Builds and compiles the LSTM model architecture.
        
        Returns:
        - Sequential: Compiled LSTM model.
        """
        model = Sequential([
            LSTM(50, input_shape=self.input_shape, return_sequences=True),
            LSTM(50, return_sequences=False),
            Dense(25, activation="relu"),
            Dense(1, activation="linear")
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        self.logger.info("LSTM model built and compiled.")
        return model

    def train(self, X_train, y_train, epochs=20, batch_size=32):
        """
        Trains the LSTM model on the provided dataset.

        Args:
        - X_train (np.ndarray): Training input data.
        - y_train (np.ndarray): Training target data.
        - epochs (int): Number of epochs to train.
        - batch_size (int): Batch size for training.
        """
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        self.model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        self.logger.info("LSTM model trained on data.")

    def predict(self, X_test) -> np.ndarray:
        """
        Makes predictions on the provided test data.

        Args:
        - X_test (np.ndarray): Test data for prediction.

        Returns:
        - np.ndarray: Predictions.
        """
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        predictions = self.model.predict(X_test_scaled)
        self.logger.info("Generated predictions using LSTM model.")
        return predictions

    def multi_step_predict(self, X_test, steps=5) -> np.ndarray:
        """
        Performs multi-step forecasting, predicting multiple future steps.

        Args:
        - X_test (np.ndarray): Test data for the initial state.
        - steps (int): Number of future steps to predict.

        Returns:
        - np.ndarray: Array of multi-step predictions.
        """
        predictions = []
        last_sequence = X_test[-1].reshape(1, *self.input_shape)
        
        for _ in range(steps):
            next_prediction = self.model.predict(last_sequence)
            predictions.append(next_prediction[0, 0])
            last_sequence = np.append(last_sequence[:, 1:], next_prediction.reshape(1, 1, 1), axis=1)
        
        self.logger.info(f"Performed multi-step prediction for {steps} steps.")
        return np.array(predictions)

    def update_model(self, X_train, y_train):
        """
        Retrains the model on new data, enabling continuous learning.

        Args:
        - X_train (np.ndarray): New training input data.
        - y_train (np.ndarray): New training target data.
        """
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        self.model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, verbose=1)
        self.logger.info("LSTM model updated with new training data.")
