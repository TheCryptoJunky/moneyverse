# all_logging/centralized_logger.py

import logging
from logging.handlers import RotatingFileHandler

class CentralizedLogger:
    def __init__(self, name="centralized", log_file="logs/centralized.log", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create handlers for rotating file logs
        handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def log(self, level, message):
        self.logger.log(level, message)
