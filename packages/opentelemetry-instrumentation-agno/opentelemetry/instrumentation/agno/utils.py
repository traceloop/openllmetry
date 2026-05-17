import logging
from typing import Any
from functools import wraps
import asyncio
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


def dont_throw(func):
    """Decorator to prevent exceptions from being thrown."""
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.debug(f"Error in {func.__name__}: {e}")
        return async_wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.debug(f"Error in {func.__name__}: {e}")
        return wrapper


def set_span_attribute(span: Span, name: str, value: Any) -> None:
    """Set an attribute on a span if the value is not None."""
    if value is not None:
        if isinstance(value, dict):
            for key, val in value.items():
                set_span_attribute(span, f"{name}.{key}", val)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                set_span_attribute(span, f"{name}.{index}", item)
        else:
            span.set_attribute(name, value)


def should_send_prompts() -> bool:
    """Check if prompts should be sent based on environment variables."""
    import os
    return os.getenv("TRACELOOP_TRACE_CONTENT", "true").lower() == "true"
