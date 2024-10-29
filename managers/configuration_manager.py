# Full file path: /moneyverse/utils/configuration_manager.py

import mysql.connector
from all_logging.centralized_logger import CentralizedLogger

logger = CentralizedLogger()

class ConfigurationManager:
    """
    Manages configuration parameters stored in the database for dynamic control.
    """

    def __init__(self):
        self.connection_params = {
            "host": "localhost",
            "user": "username",
            "password": "password",
            "database": "wallet_db"
        }

    def get_config(self, key):
        """Retrieve a configuration value by key."""
        try:
            connection = mysql.connector.connect(**self.connection_params)
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT config_value FROM configurations WHERE config_key = %s", (key,))
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            return result["config_value"] if result else None
        except mysql.connector.Error as e:
            logger.error(f"Error retrieving configuration for {key}: {e}")
            return None

    def set_config(self, key, value):
        """Update or insert a configuration value."""
        try:
            connection = mysql.connector.connect(**self.connection_params)
            cursor = connection.cursor()
            cursor.execute(
                "REPLACE INTO configurations (config_key, config_value) VALUES (%s, %s)", 
                (key, value)
            )
            connection.commit()
            cursor.close()
            connection.close()
            logger.info(f"Configuration for {key} set to {value}.")
        except mysql.connector.Error as e:
            logger.error(f"Error setting configuration for {key}: {e}")

    def delete_config(self, key):
        """Delete a configuration by key."""
        try:
            connection = mysql.connector.connect(**self.connection_params)
            cursor = connection.cursor()
            cursor.execute("DELETE FROM configurations WHERE config_key = %s", (key,))
            connection.commit()
            cursor.close()
            connection.close()
            logger.info(f"Configuration for {key} deleted.")
        except mysql.connector.Error as e:
            logger.error(f"Error deleting configuration for {key}: {e}")
