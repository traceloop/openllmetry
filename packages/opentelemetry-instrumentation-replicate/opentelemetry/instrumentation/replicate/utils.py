import logging
import traceback

from opentelemetry.semconv_ai.utils import should_send_prompts as should_send_prompts
from opentelemetry.instrumentation.replicate.config import Config

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


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


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """

    return not Config.use_legacy_attributes
