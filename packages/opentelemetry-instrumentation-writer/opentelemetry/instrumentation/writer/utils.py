import logging
import os
import traceback

from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry import context as context_api
from opentelemetry.instrumentation.writer import Config

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_SYSTEM_WRITER = "writer"

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def set_span_attribute(span, name, value):
    if value is not None and value != "":
        span.set_attribute(name, value)


def should_send_prompts():
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
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
def error_metrics_attributes(exception):
    return {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_WRITER,
        "error.type": exception.__class__.__name__,
        "error.message": str(exception),
    }


@dont_throw
def response_attributes(response):
    return {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_WRITER,
        SpanAttributes.LLM_RESPONSE_MODEL: response.get(
            "model"
        ),  # TODO check if response is a dict, not a class instance
    }


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """

    return not Config.use_legacy_attributes
