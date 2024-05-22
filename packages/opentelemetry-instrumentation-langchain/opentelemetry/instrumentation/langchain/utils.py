import json
import logging
import os
from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.semconv.ai import SpanAttributes


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


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
            logger.warning("Failed to execute %s, error: %s", func.__name__, str(e))
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


@dont_throw
def process_request(span, args, kwargs):
    if should_send_prompts():
        kwargs_to_serialize = kwargs.copy()
        for arg in args:
            if arg and isinstance(arg, dict):
                for key, value in arg.items():
                    kwargs_to_serialize[key] = value

        args = [arg for arg in args if not isinstance(arg, dict)]

        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(
                {
                    "args": [_convert_to_string(arg) for arg in args],
                    "kwargs": {
                        key: value.to_json() if hasattr(value, "to_json") else value
                        for key, value in kwargs_to_serialize.items()
                        if key != "callbacks"
                    },
                }
            ),
        )


@dont_throw
def process_response(span, response):
    if should_send_prompts():
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _convert_to_string(response),
        )


def _convert_to_string(value):
    try:
        if hasattr(value, "to_json"):
            return json.dumps(value.to_json())
        if hasattr(value, "to_string"):
            return value.to_string()

        if isinstance(value, str):
            return value

        return json.dumps(value)
    except TypeError:
        return str(value)
