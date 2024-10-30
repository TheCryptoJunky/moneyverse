# all_logging/error_logger.py

import logging
from .centralized_logger import CentralizedLogger

class ErrorLogger(CentralizedLogger):
    def __init__(self, log_file="logs/errors.log"):
        super().__init__(name="error", log_file=log_file, level=logging.ERROR)

    def log_error(self, error_message):
        self.logger.error(f"Error: {error_message}")

    def log_warning(self, warning_message):
        self.logger.warning(f"Warning: {warning_message}")
