import inspect
import logging
import traceback
import os
from opentelemetry.instrumentation.milvus.config import Config


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


def get_call_arguments(wrapped, args, kwargs):
    """
    Returns the arguments of a wrapped call as a single name -> value dict.

    SDK methods are commonly called positionally (for example
    ``client.search("collection", data)``), so reading ``kwargs`` alone would
    silently drop those arguments. Extra keyword arguments collected by a
    ``**kwargs`` parameter are flattened into the result. Falls back to
    ``kwargs`` if the signature cannot be bound.
    """
    try:
        signature = inspect.signature(wrapped)
        bound = signature.bind_partial(*args, **kwargs)
        call_args = {}
        for name, value in bound.arguments.items():
            kind = signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                call_args.update(value)
            elif kind is not inspect.Parameter.VAR_POSITIONAL:
                call_args[name] = value
        return call_args
    except Exception:
        return dict(kwargs)
