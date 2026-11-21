import logging
import os
import traceback

from opentelemetry import context as context_api
from opentelemetry.instrumentation.transformers.config import Config

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"

# The pipeline input is a required argument whose name differs between pipeline
# classes (``text_inputs`` on TextGenerationPipeline, ``inputs`` on Pipeline).
PIPELINE_INPUT_KWARGS = ("text_inputs", "inputs")


def resolve_prompts(args, kwargs):
    """Return the pipeline input, whether it was passed positionally or by keyword."""
    if args and len(args) > 0:
        return args[0]

    for key in PIPELINE_INPUT_KWARGS:
        if key in kwargs:
            return kwargs[key]

    return None


def should_send_prompts():
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


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


def should_emit_events():
    return not Config.use_legacy_attributes
