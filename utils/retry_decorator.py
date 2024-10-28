# Full file path: /moneyverse/utils/retry_decorator.py

import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry(retries=3, delay=1, backoff=2, fallback_function=None):
    """
    A decorator to retry a function upon failure.

    Parameters:
        retries (int): Number of retry attempts.
        delay (int): Initial delay between retries (in seconds).
        backoff (int): Factor to increase delay after each retry.
        fallback_function (callable): Optional function to call if all retries fail.
    """
    def decorator_retry(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay

            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__} with error: {e}")
                    attempt += 1
                    time.sleep(current_delay)
                    current_delay *= backoff  # Incrementally increase delay

            # All retries failed
            logger.error(f"All {retries} attempts failed for {func.__name__}.")
            if fallback_function:
                logger.info(f"Executing fallback for {func.__name__}.")
                return fallback_function(*args, **kwargs)
            else:
                raise RuntimeError(f"{func.__name__} failed after {retries} retries.")

        return wrapper
    return decorator_retry
