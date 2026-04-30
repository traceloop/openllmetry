import logging
import pymilvus
import traceback
import os

from opentelemetry.instrumentation.milvus.config import Config


def pymilvus_supports_async_milvus_client():
    """True when pymilvus is new enough and exposes ``AsyncMilvusClient``.

    @note: AsyncMilvusClient was added in pymilvus>=2.6.0.
    """
    return hasattr(pymilvus, "AsyncMilvusClient")


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


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"
