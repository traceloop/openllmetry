import logging
import traceback
from opentelemetry.instrumentation.alephalpha.config import Config

def dont_throw(func):
    """
    A decorator that wraps the given function and logs any exceptions instead of raising them.
    This helps maintain function execution without errors disrupting the application flow.

    @param func: The function to wrap
    @return: The wrapper function that logs any caught exceptions
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)
    logger.setLevel(logging.DEBUG)  # Set to desired level (e.g., DEBUG, INFO)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = (
                f"OpenLLMetry failed to trace in {func.__name__}, error: {traceback.format_exc()}"
            )
            logger.debug(error_message)
            if Config.exception_logger:
                # Log the error using the custom exception logger if available
                Config.exception_logger.error(error_message)

    return wrapper
