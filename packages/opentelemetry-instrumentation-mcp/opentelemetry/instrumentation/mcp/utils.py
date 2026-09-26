"""Shared utilities for MCP instrumentation."""

import asyncio
import logging
import os
import traceback


class Config:
    exception_logger = None


# Generic, content-free status description used when TRACELOOP_TRACE_CONTENT is
# disabled, so error spans still convey that a call failed without leaking the
# tool's error message or arguments.
SUPPRESSED_ERROR_DESCRIPTION = "error (content suppressed by TRACELOOP_TRACE_CONTENT)"


def should_send_prompts() -> bool:
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true"


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
