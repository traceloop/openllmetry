import logging
import os
import traceback

from opentelemetry import context as context_api
from opentelemetry.instrumentation.azure_search.config import Config

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def _is_truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def should_send_content() -> bool:
    env_setting = os.getenv(TRACELOOP_TRACE_CONTENT, "true")
    override = context_api.get_value("override_enable_content_tracing")
    if override is not None:
        return _is_truthy(override)
    return _is_truthy(env_setting)


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

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
