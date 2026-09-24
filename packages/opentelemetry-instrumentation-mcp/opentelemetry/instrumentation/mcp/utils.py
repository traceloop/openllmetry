"""Shared utilities for MCP instrumentation."""

import asyncio
import logging
import os
import traceback

from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Status, StatusCode


class Config:
    exception_logger = None


def should_send_prompts() -> bool:
    """Whether request/response content may be recorded on spans.

    Mirrors the traceloop SDK's ``TRACELOOP_TRACE_CONTENT`` switch: content
    capture is on unless an operator explicitly turns it off. Shared by the
    FastMCP server wrapper and the MCP client path so a single environment
    variable governs both, which is what the package README documents.
    """
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true"


def error_status(description: str) -> Status:
    """An ERROR status, carrying `description` only when content capture is on.

    The status code is not content, but its description is: on the client path
    it is the server's own error text. Callers record the exception type
    separately, so the failure stays visible either way.
    """
    if should_send_prompts():
        return Status(StatusCode.ERROR, description)
    return Status(StatusCode.ERROR)


def record_error(span, exc) -> None:
    """Mark `span` failed, withholding only the parts of `exc` that are content.

    The exception type and the stack frames are not content and are recorded
    either way -- the stacktrace is the part you debug from. The message is,
    and it reaches a traceback twice over: as the last line of a formatted
    one, and inside the source line of the raise site's own frame. Hence the
    frames are rendered by hand, without either.

    Both branches go through ``record_exception`` so the event's non-content
    fields (``exception.type``, ``exception.escaped``) are written by the SDK
    and match whichever way the switch is set; only the content fields are
    overridden.
    """
    span.set_attribute(ERROR_TYPE, type(exc).__name__)
    if should_send_prompts():
        span.record_exception(exc)
    else:
        span.record_exception(
            exc,
            attributes={
                "exception.message": "",
                # File, line and function, but not the frame's source line: a
                # raise site like ToolError("...") reproduces its own message
                # there, and that message is the thing being withheld.
                "exception.stacktrace": "\n".join(
                    f'  File "{frame.filename}", line {frame.lineno},'
                    f" in {frame.name}"
                    for frame in traceback.extract_tb(exc.__traceback__)
                ),
            },
        )
    span.set_status(error_status(str(exc)))


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    Works for both synchronous and asynchronous functions.
    """
    logger = logging.getLogger(func.__module__)

    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def _handle_exception(e, func, logger):
        logger.debug(
            "OpenLLMetry failed to trace in %s, error: %s",
            func.__name__,
            traceback.format_exc(),
        )
        if Config.exception_logger:
            Config.exception_logger(e)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
