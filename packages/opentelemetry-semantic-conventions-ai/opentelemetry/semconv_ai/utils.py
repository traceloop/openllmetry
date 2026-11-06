import logging
import os

from opentelemetry import context as context_api

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)
TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"

# Values defined by the OTel GenAI conventions for the capture switch. Any
# other value is parsed as a boolean, so ``false`` also disables capture.
_OTEL_CAPTURE_VALUES = {
    "no_content": False,
    "span_only": True,
    "event_only": True,
    "span_and_event": True,
}


def _is_truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def should_send_prompts() -> bool:
    """Return whether prompts and completions may be recorded.

    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is honored when it
    is explicitly set; otherwise ``TRACELOOP_TRACE_CONTENT`` is used, which
    defaults to on. The ``override_enable_content_tracing`` context value can
    still enable capture for a specific call.
    """
    configured = os.getenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT)
    if configured is not None:
        normalized = configured.strip().lower()
        if normalized in _OTEL_CAPTURE_VALUES:
            enabled = _OTEL_CAPTURE_VALUES[normalized]
        else:
            enabled = _is_truthy(configured)
    else:
        enabled = _is_truthy(os.getenv(TRACELOOP_TRACE_CONTENT, "true"))

    return enabled or _is_truthy(
        context_api.get_value("override_enable_content_tracing")
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
            logger.warning(
                "Failed to execute %s, error: %s", func.__name__, str(e)
            )  # TODO: posthog instead
            logger.warning(
                "Failed to set attributes for openai span, error: %s", str(e)
            )

    return wrapper
