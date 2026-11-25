import logging
import os
import traceback
from enum import Enum

from opentelemetry import context as context_api
from opentelemetry.instrumentation.together.config import Config
from opentelemetry.semconv_ai import GenAISystem

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"

# OTel's GenAiProviderNameValues has no Together entry, so use this repo's constant.
TOGETHER_PROVIDER = GenAISystem.TOGETHER_AI.value


def set_span_attribute(span, name, value):
    if value is not None and value != "":
        span.set_attribute(name, value)


def enum_value(value):
    """together 1.x returns str enums (e.g. FinishReason.EOS); 2.x/3.x returns plain strings."""
    return value.value if isinstance(value, Enum) else value


def model_as_dict(model):
    """Return SDK models (pydantic in every together version) as plain dicts."""
    if isinstance(model, dict):
        return model
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


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


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """
    return not Config.use_legacy_attributes
