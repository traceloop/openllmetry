import os
from typing import Optional


def is_tracing_enabled() -> bool:
    return (os.getenv("TRACELOOP_TRACING_ENABLED") or "true").lower() == "true"


def is_content_tracing_enabled(trace_content: Optional[bool] = None) -> bool:
    if trace_content is not None:
        return trace_content
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true"


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def is_logging_enabled() -> bool:
    return (os.getenv("TRACELOOP_LOGGING_ENABLED") or "false").lower() == "true"
