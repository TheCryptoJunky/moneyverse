import logging
import time
from typing import Callable, Any

class FaultTolerance:
    """
    Manages fault tolerance across system components, including error logging and retry mechanisms.

    Attributes:
    - max_retries (int): Maximum number of retries for a failed function.
    - retry_delay (float): Delay between retries in seconds.
    - logger (Logger): Logger for tracking errors and recovery attempts.
    """

    def __init__(self, max_retries=3, retry_delay=2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        self.logger.info("FaultTolerance module initialized.")

    def execute_with_retries(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executes a function with automatic retries in case of failure.

        Args:
        - func (Callable): Function to execute.
        - args: Positional arguments for the function.
        - kwargs: Keyword arguments for the function.

        Returns:
        - Any: Result of the function if successful, None if all retries fail.
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                result = func(*args, **kwargs)
                self.logger.info(f"Function {func.__name__} executed successfully.")
                return result
            except Exception as e:
                attempt += 1
                self.logger.error(f"Error in {func.__name__}: {e}. Attempt {attempt}/{self.max_retries}. Retrying in {self.retry_delay} seconds.")
                time.sleep(self.retry_delay)
        
        self.logger.critical(f"Function {func.__name__} failed after {self.max_retries} attempts.")
        return None

    def monitor_component(self, component_name: str, health_check_func: Callable, recovery_func: Callable):
        """
        Monitors a component's health, triggering recovery if health check fails.

        Args:
        - component_name (str): Name of the component to monitor.
        - health_check_func (Callable): Function that returns True if healthy, False otherwise.
        - recovery_func (Callable): Function to attempt recovery if health check fails.
        """
        if not health_check_func():
            self.logger.warning(f"Health check failed for {component_name}. Attempting recovery.")
            if self.execute_with_retries(recovery_func):
                self.logger.info(f"Recovery successful for {component_name}.")
            else:
                self.logger.critical(f"Failed to recover {component_name}.")

    def handle_transaction_error(self, transaction_func: Callable, *args, **kwargs):
        """
        Handles errors specific to transactions, with detailed logging and recovery.

        Args:
        - transaction_func (Callable): Transaction function to execute.
        - args: Positional arguments for the transaction function.
        - kwargs: Keyword arguments for the transaction function.

        Returns:
        - Any: Result of the transaction function if successful, None otherwise.
        """
        try:
            result = transaction_func(*args, **kwargs)
            self.logger.info(f"Transaction executed successfully in {transaction_func.__name__}.")
            return result
        except Exception as e:
            self.logger.error(f"Transaction error in {transaction_func.__name__}: {e}")
            self.execute_with_retries(transaction_func, *args, **kwargs)
            self.logger.warning(f"Attempted recovery for transaction in {transaction_func.__name__}.")

    def log_and_raise(self, message: str):
        """
        Logs a critical error and raises an exception.

        Args:
        - message (str): Error message to log and raise.
        """
        self.logger.critical(message)
        raise Exception(message)
