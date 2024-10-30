import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from ..database.db_connection import DatabaseConnection

class TrendForecaster:
    """
    Forecasts market trends using ARIMA and regression modeling with adaptive multi-horizon forecasting.
    
    Attributes:
    - db (DatabaseConnection): Database for logging predictions.
    - model_cache (dict): Cache of trained models by asset.
    - params (dict): Parameters for ARIMA and linear regression models.
    """

    def __init__(self, db: DatabaseConnection, params=None):
        self.db = db
        self.params = params or {"arima_order": (5, 1, 0), "regression_weights": [0.8, 0.2]}
        self.model_cache = {}
        self.logger = logging.getLogger(__name__)

    def train_arima_model(self, asset: str, historical_data: np.ndarray):
        """
        Trains an ARIMA model for a specified asset.

        Args:
        - asset (str): Asset symbol.
        - historical_data (np.ndarray): Historical price data.
        """
        model = ARIMA(historical_data, order=self.params["arima_order"])
        self.model_cache[asset] = model.fit()
        self.logger.info(f"Trained ARIMA model for {asset}.")

    def forecast_arima(self, asset: str, steps: int = 5) -> np.ndarray:
        """
        Generates a forecast using the ARIMA model.

        Args:
        - asset (str): Asset symbol.
        - steps (int): Forecast horizon.
        
        Returns:
        - np.ndarray: Forecasted values.
        """
        arima_model = self.model_cache.get(asset)
        if not arima_model:
            self.logger.warning(f"No trained ARIMA model found for {asset}.")
            return np.array([])
        
        forecast = arima_model.forecast(steps=steps)
        self.logger.info(f"ARIMA forecast for {asset}: {forecast}")
        return forecast

    def regression_forecast(self, recent_data: np.ndarray, steps: int = 5) -> np.ndarray:
        """
        Generates a linear regression forecast based on recent data trends.

        Args:
        - recent_data (np.ndarray): Recent price data.
        - steps (int): Forecast horizon.

        Returns:
        - np.ndarray: Forecasted values.
        """
        model = LinearRegression()
        x = np.arange(len(recent_data)).reshape(-1, 1)
        model.fit(x, recent_data)
        x_future = np.arange(len(recent_data), len(recent_data) + steps).reshape(-1, 1)
        forecast = model.predict(x_future)
        self.logger.info(f"Regression forecast: {forecast}")
        return forecast

    def ensemble_forecast(self, arima_forecast: np.ndarray, regression_forecast: np.ndarray) -> np.ndarray:
        """
        Combines forecasts from ARIMA and regression using weighted averaging.

        Args:
        - arima_forecast (np.ndarray): ARIMA model forecast.
        - regression_forecast (np.ndarray): Regression model forecast.

        Returns:
        - np.ndarray: Ensemble forecast.
        """
        weights = self.params["regression_weights"]
        ensemble = weights[0] * arima_forecast + weights[1] * regression_forecast
        self.logger.info(f"Ensemble forecast: {ensemble}")
        return ensemble

    def update_forecasts(self, asset: str, historical_data: np.ndarray, recent_data: np.ndarray, steps: int = 5):
        """
        Trains models, generates an ensemble forecast, and logs results.

        Args:
        - asset (str): Asset symbol.
        - historical_data (np.ndarray): Historical data for ARIMA training.
        - recent_data (np.ndarray): Recent data for regression and ensemble.
        - steps (int): Forecast horizon.
        """
        self.train_arima_model(asset, historical_data)
        arima_forecast = self.forecast_arima(asset, steps)
        regression_forecast = self.regression_forecast(recent_data, steps)
        ensemble_forecast = self.ensemble_forecast(arima_forecast, regression_forecast)
        self.db.log_forecast(asset, ensemble_forecast.tolist())
        self.logger.info(f"Forecast for {asset} logged successfully.")
