# File: /src/logging/logger-config.py

import all_logging
import sys
from logging.handlers import RotatingFileHandler
from centralized_logger import CentralizedLogger

def setup_logger(log_level=logging.INFO, log_file=None):
    """Sets up logging configuration with both file rotation and MySQL integration."""
    logger = logging.getLogger()

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # Console handler for real-time output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optionally add file handler with rotation
    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Initialize the centralized logger (includes MySQL)
    centralized_logger = CentralizedLogger()

    return logger, centralized_logger
