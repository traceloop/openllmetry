import asyncio
import dataclasses
import json
import logging
import os
import traceback
from opentelemetry import context as context_api


def set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _is_truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def should_send_prompts():
    env_setting = os.getenv("TRACELOOP_TRACE_CONTENT", "true")
    override = context_api.get_value("override_enable_content_tracing")
    return _is_truthy(env_setting) or bool(override)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if hasattr(o, "to_json"):
            return o.to_json()

        if hasattr(o, "model_dump_json"):
            return o.model_dump_json()
        elif hasattr(o, "json"):
            return o.json()

        if hasattr(o, "__class__"):
            return o.__class__.__name__

        return super().default(o)


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

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
