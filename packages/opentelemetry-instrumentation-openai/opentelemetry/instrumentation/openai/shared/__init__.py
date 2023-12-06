import os
from opentelemetry import context as context_api


def should_send_prompts():
    return (
            os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")
