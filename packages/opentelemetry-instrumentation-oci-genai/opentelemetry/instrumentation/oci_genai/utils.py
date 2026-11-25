import logging
import os
import re
import traceback
from urllib.parse import urlparse

from opentelemetry import context as context_api
from opentelemetry.instrumentation.oci_genai.config import Config

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"

# The OCI SDK keeps unresolved realm/dual-stack placeholders such as ``{dualStack?ds.:}`` in
# ``base_client.endpoint`` until the request is actually built; strip them for ``server.address``.
_ENDPOINT_TEMPLATE_RE = re.compile(r"\{[^}]*\}")


def set_span_attribute(span, name, value):
    if value is not None and value != "":
        span.set_attribute(name, value)


def should_send_prompts():
    return (os.getenv(TRACELOOP_TRACE_CONTENT) or "true").lower() == "true" or context_api.get_value(
        "override_enable_content_tracing"
    )


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


def model_as_dict(model):
    """Convert an OCI SDK model (or a container of models) into plain Python data."""
    if model is None or isinstance(model, (str, int, float, bool)):
        return model
    if isinstance(model, dict):
        return {key: model_as_dict(value) for key, value in model.items()}
    if isinstance(model, (list, tuple)):
        return [model_as_dict(item) for item in model]
    try:
        from oci.util import to_dict

        return to_dict(model)
    except Exception:
        return getattr(model, "__dict__", str(model))


def get_server_address(instance):
    """Return the hostname of the OCI GenAI inference endpoint used by ``instance``."""
    endpoint = getattr(getattr(instance, "base_client", None), "endpoint", None)
    if not isinstance(endpoint, str) or not endpoint:
        return None
    endpoint = _ENDPOINT_TEMPLATE_RE.sub("", endpoint)
    parsed = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
    return parsed.hostname
