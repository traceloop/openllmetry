import inspect
import logging
import traceback
from opentelemetry.instrumentation.lancedb.config import Config


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


def get_call_arguments(wrapped, args, kwargs):
    """
    Bind ``args`` and ``kwargs`` to the wrapped method's signature and return
    them as a single name -> value dict.

    Uses ``inspect.signature(wrapped).bind_partial`` so arguments passed
    positionally (for example ``table.add(data)``) are captured by parameter
    name instead of being silently dropped. Extra keyword arguments collected
    by a ``**kwargs`` parameter are flattened into the result. Falls back to
    the original ``kwargs`` if inspection or binding fails.
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
