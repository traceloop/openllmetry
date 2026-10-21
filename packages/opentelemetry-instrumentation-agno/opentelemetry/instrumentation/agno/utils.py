import logging
import traceback
from contextlib import contextmanager
from typing import Any, Iterator
from functools import wraps
import asyncio
from opentelemetry.instrumentation.agno.config import Config
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


@contextmanager
def guard_trace(logger: logging.Logger, origin: str) -> Iterator[None]:
    """Run instrumentation-only code, logging rather than raising its failures.

    Use this in a wrapper that calls into the traced library: guard the span
    work, never the wrapped call itself. A failure in this instrumentation is
    logged (and offered to ``Config.exception_logger``) while the traced
    program carries on; an exception raised by the wrapped call reaches the
    caller unchanged. Only ``Exception`` is swallowed, so cancellation keeps
    unwinding.
    """
    try:
        yield
    except Exception as e:
        logger.debug(
            "OpenLLMetry failed to trace in %s, error: %s",
            origin,
            traceback.format_exc(),
        )
        if Config.exception_logger:
            Config.exception_logger(e)


def dont_throw(func):
    """Decorator to prevent exceptions from being thrown.

    Only sound for a function that is instrumentation throughout. Do not put it
    on a wrapper that calls into the traced library: it turns that library's
    failure into a silent ``None``. Guard the span work with ``guard_trace``
    and leave the wrapped call outside the guard instead.
    """
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
