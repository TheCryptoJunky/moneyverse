# File: /src/utils/error_handler.py

import logging
import traceback
from centralized_logger import CentralizedLogger

# Initialize logger and centralized logging
logger = logging.getLogger(__name__)
centralized_logger = CentralizedLogger()

class ErrorHandler:
    """
    Centralized error handler that logs all system errors across bots, AI agents, and helpers.
    Provides a unified mechanism for tracking and analyzing errors in real-time.
    """

    def log_error(self, error_message, exception):
        """
        Logs the error message and exception details, including stack trace, to centralized logging.
        """
        error_details = {
            'message': error_message,
            'exception': str(exception),
            'stack_trace': traceback.format_exc()
        }
        logger.error(f"Error occurred: {error_message} - {exception}")
        centralized_logger.log_event(f"Error occurred: {error_message}", error_details)

    def handle_exception(self, bot_name, exception):
        """
        Handles an exception thrown by any bot and logs the error for centralized tracking.
        """
        error_message = f"Exception occurred in {bot_name}: {str(exception)}"
        self.log_error(error_message, exception)
