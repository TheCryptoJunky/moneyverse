# all_logging/trade_logger.py

import logging
from .centralized_logger import CentralizedLogger

class TradeLogger(CentralizedLogger):
    def __init__(self, log_file="logs/trades.log"):
        super().__init__(name="trade", log_file=log_file, level=logging.INFO)

    def log_trade(self, trade_id, asset, amount, price, status="executed"):
        self.logger.info(f"Trade ID: {trade_id} | Asset: {asset} | Amount: {amount} | Price: {price} | Status: {status}")
