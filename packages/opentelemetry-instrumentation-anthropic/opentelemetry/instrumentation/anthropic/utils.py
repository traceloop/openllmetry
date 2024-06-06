import os
import logging
import traceback
from opentelemetry import context as context_api
from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.semconv.ai import SpanAttributes

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_SYSTEM_ANTHROPIC = "anthropic"


def set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


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
def shared_metrics_attributes(response):
    if not isinstance(response, dict):
        response = response.__dict__

    return {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_ANTHROPIC,
        SpanAttributes.LLM_RESPONSE_MODEL: response.get("model"),
    }


@dont_throw
def error_metrics_attributes(exception):
    return {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_ANTHROPIC,
        "error.type": exception.__class__.__name__,
    }
