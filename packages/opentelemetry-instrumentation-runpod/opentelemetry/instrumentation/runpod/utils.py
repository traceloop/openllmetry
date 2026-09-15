import json
import logging
import os
import traceback

from opentelemetry import context as context_api
from opentelemetry.instrumentation.runpod.config import Config

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def dont_throw(func):
    """
    Decorator that wraps the passed in function and logs exceptions instead of throwing them.

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


def should_send_prompts():
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """
    return not Config.use_legacy_attributes


def dump_object(obj):
    """Serializes an arbitrary object to a JSON string, never raising."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def get_request_input(args, kwargs):
    """
    Returns the payload passed to ``Endpoint.run`` / ``Endpoint.run_sync``.

    The RunPod SDK accepts it positionally or as the ``request_input`` keyword.
    """
    if "request_input" in kwargs:
        return kwargs.get("request_input")
    if args:
        return args[0]
    return None


def unwrap_input(request_input):
    """
    Mirrors the normalization the RunPod SDK performs before POSTing a job: the
    payload is sent inside an ``input`` key unless it already contains one.
    """
    if isinstance(request_input, dict) and not request_input.get("input"):
        return {"input": request_input}
    return request_input


def is_runpod_job(obj) -> bool:
    """
    Returns True for a ``Job`` handle returned by the SDK.

    ``Endpoint.run``/``AsyncioEndpoint.run`` always return one, and
    ``Endpoint.run_sync`` returns the ``Job.output()`` result rather than the
    handle itself, so this is used by the ``run`` spans and as a safety net for
    the ``run_sync`` span.
    """
    return hasattr(obj, "job_id") and hasattr(obj, "endpoint_id")
