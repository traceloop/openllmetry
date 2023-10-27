import os


def is_tracing_enabled() -> bool:
    return (os.getenv("TRACELOOP_TRACING_ENABLED") or "true").lower() == "true"


def is_content_tracing_enabled() -> bool:
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true"
