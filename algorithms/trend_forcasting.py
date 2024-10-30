import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from ..database.db_connection import DatabaseConnection

class TrendForecaster:
    """
    Forecasts market trends using ARIMA modeling and integrates multi-horizon forecasting.
    
    Attributes:
    - db (DatabaseConnection): Database connection for logging predictions.
    - model_params (dict): Parameters for ARIMA model initialization.
    """

    def __init__(self, db: DatabaseConnection, model_params=None):
        self.db = db
        self.model_params = model_params or {"order": (5, 1, 0)}
        self.model_cache = {}
        self.logger = logging.getLogger(__name__)

    def train_model(self, asset: str, historical_data: np.ndarray):
        """
        Trains an ARIMA model for a given asset with historical data.
        
        Args:
        - asset (str): Asset symbol to train model on.
        - historical_data (np.ndarray): Historical price data.
        """
        model = ARIMA(historical_data, order=self.model_params["order"])
        self.model_cache[asset] = model.fit()
        self.logger.info(f"ARIMA model trained for {asset}.")

    def forecast(self, asset: str, steps: int = 5) -> np.ndarray:
        """
        Generates a multi-horizon forecast for the specified asset.
        
        Args:
        - asset (str): Asset symbol to forecast.
        - steps (int): Number of time steps to forecast.
        
        Returns:
        - np.ndarray: Array of forecasted values.
        """
        model = self.model_cache.get(asset)
        if not model:
            self.logger.warning(f"No model found for {asset}.")
            return np.array([])

        forecast = model.forecast(steps=steps)
        self.logger.info(f"Generated forecast for {asset}: {forecast}")
        return forecast

    def update_predictions(self, asset: str, forecast: np.ndarray):
        """
        Logs predictions into the database for historical tracking and analysis.
        
        Args:
        - asset (str): Asset symbol.
        - forecast (np.ndarray): Forecasted values.
        """
        self.db.log_forecast(asset, forecast.tolist())
        self.logger.info(f"Forecast for {asset} updated in the database.")

    def analyze_trend(self, asset: str, recent_data: np.ndarray) -> str:
        """
        Analyzes recent price data to determine if a trend is bullish, bearish, or neutral.
        
        Args:
        - asset (str): Asset symbol.
        - recent_data (np.ndarray): Most recent price data.
        
        Returns:
        - str: "bullish", "bearish", or "neutral" indicating trend direction.
        """
        slope = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
        trend = "bullish" if slope > 0 else "bearish" if slope < 0 else "neutral"
        self.logger.info(f"Trend analysis for {asset}: {trend}")
        return trend

    def apply_forecast(self, asset: str, recent_data: np.ndarray, steps: int = 5):
        """
        Trains, forecasts, and logs the trend for the specified asset.

        Args:
        - asset (str): Asset symbol.
        - recent_data (np.ndarray): Recent data for model training.
        - steps (int): Forecast horizon.
        """
        self.train_model(asset, recent_data)
        forecast = self.forecast(asset, steps)
        self.update_predictions(asset, forecast)
        trend = self.analyze_trend(asset, recent_data)
        self.logger.info(f"Finalized forecast and trend for {asset}: {trend}")
