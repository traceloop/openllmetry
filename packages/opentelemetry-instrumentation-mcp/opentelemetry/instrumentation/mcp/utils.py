"""Shared utilities for MCP instrumentation."""

import asyncio
import logging
import os
import traceback

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
