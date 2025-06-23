import logging
import traceback
from typing import Optional

from opentelemetry.instrumentation.google_generativeai.config import Config
from importlib.metadata import Distribution, distributions


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

def _get_package_name(dist: Distribution) -> Optional[str]:
    try:
        return dist.name.lower()
    except (KeyError, AttributeError):
        return None
    
installed_packages = {name for dist in distributions() if (name := _get_package_name(dist)) is not None}

def is_package_installed(package_name: str) -> bool:
    return package_name.lower() in installed_packages
