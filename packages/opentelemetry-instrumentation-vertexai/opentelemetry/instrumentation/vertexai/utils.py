import asyncio
import logging
import os
import traceback

from opentelemetry import context as context_api
from opentelemetry.instrumentation.vertexai.config import Config

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def should_send_prompts():
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    Works for both synchronous and asynchronous functions.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def _handle_exception(e):
        logger.debug(
            "OpenLLMetry failed to trace in %s, error: %s",
            func.__name__,
            traceback.format_exc(),
        )
        if Config.exception_logger:
            try:
                Config.exception_logger(e)
            except Exception:
                logger.debug(
                    "OpenLLMetry exception logger failed in %s",
                    func.__name__,
                    exc_info=True,
                )

    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e)

    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def should_emit_events():
    return not Config.use_legacy_attributes
