# moneyverse/managers/configuration_manager.py

import logging
from moneyverse.database.db_connection import DatabaseConnection

class ConfigurationManager:
    """
    Manages configuration settings across strategies, risk, and operational parameters.

    Attributes:
    - db_connection (DatabaseConnection): Connection to the database for persisting configurations.
    - logger (Logger): Logs configuration changes and retrieval actions.
    """

    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        self.logger = logging.getLogger(__name__)
        self.logger.info("ConfigurationManager initialized with database connection.")

    def set_config(self, key: str, value: float):
        """
        Sets a configuration parameter.

        Args:
        - key (str): Name of the configuration setting.
        - value (float): Value of the configuration setting.
        """
        try:
            self.db_connection.update_configuration(key, value)
            self.logger.info(f"Set configuration '{key}' to {value}")
        except Exception as e:
            self.logger.error(f"Failed to set configuration '{key}': {str(e)}")

    def get_config(self, key: str) -> float:
        """
        Retrieves the value of a configuration parameter.

        Args:
        - key (str): Name of the configuration setting.

        Returns:
        - float: Value of the configuration setting.
        """
        try:
            value = self.db_connection.fetch_configuration(key)
            self.logger.info(f"Retrieved configuration '{key}': {value}")
            return value
        except Exception as e:
            self.logger.error(f"Failed to retrieve configuration '{key}': {str(e)}")
            return 0.0

    def update_risk_limits(self, max_risk_per_trade: float, max_daily_loss: float):
        """
        Updates risk limits and persists them in the database.

        Args:
        - max_risk_per_trade (float): Maximum allowed risk per trade.
        - max_daily_loss (float): Maximum allowed daily loss as a percentage.
        """
        self.set_config("max_risk_per_trade", max_risk_per_trade)
        self.set_config("max_daily_loss", max_daily_loss)
        self.logger.info("Updated risk limits.")

    def update_trade_parameters(self, trade_frequency: int, profit_target: float):
        """
        Updates trade parameters such as trade frequency and profit target.

        Args:
        - trade_frequency (int): Desired frequency of trades.
        - profit_target (float): Targeted profit percentage.
        """
        self.set_config("trade_frequency", trade_frequency)
        self.set_config("profit_target", profit_target)
        self.logger.info("Updated trade parameters.")

    def load_all_configurations(self) -> dict:
        """
        Loads all configurations from the database.

        Returns:
        - dict: Dictionary containing all configuration settings.
        """
        try:
            configurations = self.db_connection.fetch_all_configurations()
            self.logger.info("Loaded all configurations.")
            return configurations
        except Exception as e:
            self.logger.error("Failed to load configurations: " + str(e))
            return {}

    def reset_configurations(self):
        """
        Resets all configurations to default values.
        """
        try:
            self.db_connection.reset_configurations()
            self.logger.info("All configurations reset to default values.")
        except Exception as e:
            self.logger.error("Failed to reset configurations: " + str(e))
