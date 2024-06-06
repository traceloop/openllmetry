import dataclasses
import json
import logging
import os
import traceback
from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.semconv.ai import SpanAttributes


class CallbackFilteredJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dict):
            if "callbacks" in o:
                del o["callbacks"]
                return o
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if hasattr(o, "to_json"):
            return o.to_json()

        return super().default(o)


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
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
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
                    if key == "callbacks":
                        continue
                    kwargs_to_serialize[key] = value

        args_to_serialize = [arg for arg in args if not isinstance(arg, dict)]

        entity_input = {"args": args_to_serialize, "kwargs": kwargs_to_serialize}

        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(entity_input, cls=CallbackFilteredJSONEncoder),
        )


@dont_throw
def process_response(span, response):
    if should_send_prompts():
        if isinstance(response, str):
            output_entity = response
        else:
            output_entity = json.dumps(response, cls=CallbackFilteredJSONEncoder)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            output_entity,
        )
