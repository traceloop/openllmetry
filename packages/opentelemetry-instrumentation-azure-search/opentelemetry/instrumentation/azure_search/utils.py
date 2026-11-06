import logging
import os
import traceback
from functools import wraps

from opentelemetry.instrumentation.azure_search.config import Config


def dont_throw(func):
    """Decorator that wraps the passed in function and logs exceptions instead of throwing them.

    Args:
        func: The function to wrap.

    Returns:
        The wrapper function.
    """
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


def set_span_attribute(span, name, value):
    if value is not None and value != "":
        span.set_attribute(name, value)


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"
