"""Utility functions for Together AI instrumentation."""

from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from opentelemetry.trace import Status, StatusCode

T = TypeVar('T')

def dont_throw(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    """Decorator that prevents functions from throwing exceptions.
    
    Instead of throwing, it logs the error and returns None.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            if 'span' in kwargs:
                span = kwargs['span']
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(exc)
            return None
    return wrapper

def extract_model_name(response: Dict[str, Any]) -> Optional[str]:
    """Extract model name from Together AI response.
    
    Args:
        response: The response from Together AI API
        
    Returns:
        The model name if found, None otherwise
    """
    try:
        if isinstance(response, dict):
            return response.get('model', None)
        return None
    except (AttributeError, KeyError):
        return None

def extract_completion_tokens(response: Dict[str, Any]) -> Optional[int]:
    """Extract completion tokens from Together AI response.
    
    Args:
        response: The response from Together AI API
        
    Returns:
        The number of completion tokens if found, None otherwise
    """
    try:
        if isinstance(response, dict) and 'usage' in response:
            return response['usage'].get('completion_tokens', None)
        return None
    except (AttributeError, KeyError):
        return None
