import os

class Config:
    exception_logger = None
    use_legacy_attributes = True  # Default to legacy behavior for backward compatibility

    @staticmethod
    def is_tracing_enabled() -> bool:
        return (os.getenv("TRACELOOP_TRACING_ENABLED") or "true").lower() == "true"

    @staticmethod
    def is_content_tracing_enabled() -> bool:
        return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true"

    @staticmethod
    def is_metrics_enabled() -> bool:
        return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"

    @staticmethod
    def is_logging_enabled() -> bool:
        return (os.getenv("TRACELOOP_LOGGING_ENABLED") or "false").lower() == "true"