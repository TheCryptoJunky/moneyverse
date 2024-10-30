import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from ..database.db_connection import DatabaseConnection

class TrendForecaster:
    """
    Forecasts market trends using ARIMA and ensemble forecasting techniques.
    
    Attributes:
    - db (DatabaseConnection): Database for logging predictions.
    - model_cache (dict): Cache for storing trained ARIMA models by asset.
    - ensemble_params (dict): Parameters for ARIMA and ensemble modeling.
    """

    def __init__(self, db: DatabaseConnection, ensemble_params=None):
        self.db = db
        self.ensemble_params = ensemble_params or {"arima_order": (5, 1, 0), "ensemble_weights": [0.7, 0.3]}
        self.model_cache = {}
        self.logger = logging.getLogger(__name__)

    def train_arima_model(self, asset: str, historical_data: np.ndarray):
        """
        Trains an ARIMA model for a specified asset.

        Args:
        - asset (str): Asset to train on.
        - historical_data (np.ndarray): Historical price data.
        """
        model = ARIMA(historical_data, order=self.ensemble_params["arima_order"])
        self.model_cache[asset] = model.fit()
        self.logger.info(f"Trained ARIMA model for {asset}.")

    def forecast(self, asset: str, steps: int = 5) -> np.ndarray:
        """
        Generates a multi-horizon forecast for the specified asset.

        Args:
        - asset (str): Asset symbol to forecast.
        - steps (int): Number of time steps to forecast.

        Returns:
        - np.ndarray: Forecasted values.
        """
        arima_model = self.model_cache.get(asset)
        if not arima_model:
            self.logger.warning(f"No trained ARIMA model for {asset}.")
            return np.array([])

        forecast = arima_model.forecast(steps=steps)
        self.logger.info(f"Generated forecast for {asset}: {forecast}")
        return forecast

    def ensemble_forecast(self, arima_forecast: np.ndarray, recent_trend: np.ndarray) -> np.ndarray:
        """
        Generates an ensemble forecast by combining ARIMA and recent trend data.

        Args:
        - arima_forecast (np.ndarray): Forecasted values from ARIMA.
        - recent_trend (np.ndarray): Recent price trend data.

        Returns:
        - np.ndarray: Ensemble forecast values.
        """
        weights = self.ensemble_params["ensemble_weights"]
        ensemble_forecast = weights[0] * arima_forecast + weights[1] * recent_trend[-len(arima_forecast):]
        self.logger.info(f"Ensemble forecast generated: {ensemble_forecast}")
        return ensemble_forecast

    def update_predictions(self, asset: str, forecast: np.ndarray):
        """
        Logs predictions into the database.

        Args:
        - asset (str): Asset symbol.
        - forecast (np.ndarray): Forecasted values.
        """
        self.db.log_forecast(asset, forecast.tolist())
        self.logger.info(f"Forecast for {asset} updated in the database.")

    def apply_trend_forecast(self, asset: str, historical_data: np.ndarray, recent_trend: np.ndarray, steps: int = 5):
        """
        Trains the model, generates an ensemble forecast, and logs results.

        Args:
        - asset (str): Asset symbol.
        - historical_data (np.ndarray): Historical data for model training.
        - recent_trend (np.ndarray): Recent data for trend analysis.
        - steps (int): Forecast horizon.
        """
        self.train_arima_model(asset, historical_data)
        arima_forecast = self.forecast(asset, steps)
        ensemble_prediction = self.ensemble_forecast(arima_forecast, recent_trend)
        self.update_predictions(asset, ensemble_prediction)
        self.logger.info(f"Finalized ensemble forecast for {asset}.")
