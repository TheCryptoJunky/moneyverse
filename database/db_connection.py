import sqlite3
import logging
from typing import Any, Dict, List

class DatabaseConnection:
    """
    Manages database operations, including logging and retrieving AI model data, wallet information, and strategy performance.
    
    Attributes:
    - db_path (str): Path to the database file.
    - connection (sqlite3.Connection): Database connection instance.
    - logger (Logger): Logger for tracking database operations.
    """

    def __init__(self, db_path="moneyverse.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.connection.cursor()
        self.logger = logging.getLogger(__name__)
        self._create_tables()
        self.logger.info("DatabaseConnection initialized.")

    def _create_tables(self):
        """
        Creates necessary tables for AI models, wallets, strategies, and logging.
        """
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                performance_metric REAL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS wallet_data (
                wallet_address TEXT PRIMARY KEY,
                asset_type TEXT,
                balance REAL,
                nav REAL,
                strategy TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_model_performance (
                model_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric REAL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pgm_predictions (
                condition TEXT,
                prediction REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.connection.commit()
        self.logger.info("Database tables created or verified.")

    def log_strategy_performance(self, strategy_name: str, performance_metric: float):
        """
        Logs performance metrics for strategies.

        Args:
        - strategy_name (str): Name of the strategy.
        - performance_metric (float): Performance metric to log.
        """
        self.cursor.execute("""
            INSERT INTO strategy_performance (strategy_name, performance_metric)
            VALUES (?, ?)
        """, (strategy_name, performance_metric))
        self.connection.commit()
        self.logger.info(f"Logged performance for strategy {strategy_name}.")

    def update_wallet_data(self, wallet_address: str, asset_type: str, balance: float, nav: float, strategy: str):
        """
        Updates wallet data in the database.

        Args:
        - wallet_address (str): Address of the wallet.
        - asset_type (str): Type of asset held in the wallet.
        - balance (float): Current balance of the wallet.
        - nav (float): Net asset value of the wallet.
        - strategy (str): Strategy currently assigned to the wallet.
        """
        self.cursor.execute("""
            INSERT OR REPLACE INTO wallet_data (wallet_address, asset_type, balance, nav, strategy)
            VALUES (?, ?, ?, ?, ?)
        """, (wallet_address, asset_type, balance, nav, strategy))
        self.connection.commit()
        self.logger.info(f"Updated data for wallet {wallet_address}.")

    def log_model_performance(self, model_type: str, metric: float):
        """
        Logs performance metrics for AI models.

        Args:
        - model_type (str): Type of AI model.
        - metric (float): Performance metric to log.
        """
        self.cursor.execute("""
            INSERT INTO ai_model_performance (model_type, metric)
            VALUES (?, ?)
        """, (model_type, metric))
        self.connection.commit()
        self.logger.info(f"Logged performance for model {model_type}.")

    def log_pgm_prediction(self, condition: str, prediction: float):
        """
        Logs prediction results from the probabilistic graphical model (PGM).

        Args:
        - condition (str): Market condition.
        - prediction (float): Prediction probability.
        """
        self.cursor.execute("""
            INSERT INTO pgm_predictions (condition, prediction)
            VALUES (?, ?)
        """, (condition, prediction))
        self.connection.commit()
        self.logger.info(f"Logged PGM prediction for condition {condition}.")

    def get_strategy_performance(self, strategy_name: str) -> float:
        """
        Retrieves the most recent performance metric for a given strategy.

        Args:
        - strategy_name (str): Name of the strategy.

        Returns:
        - float: Most recent performance metric for the strategy.
        """
        self.cursor.execute("""
            SELECT performance_metric FROM strategy_performance
            WHERE strategy_name = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (strategy_name,))
        result = self.cursor.fetchone()
        return result[0] if result else 0.0

    def get_wallets(self) -> List[Dict[str, Any]]:
        """
        Retrieves all wallet data.

        Returns:
        - list: List of dictionaries with wallet data.
        """
        self.cursor.execute("SELECT * FROM wallet_data")
        wallets = [{"wallet_address": row[0], "asset_type": row[1], "balance": row[2], "nav": row[3], "strategy": row[4]} for row in self.cursor.fetchall()]
        return wallets
