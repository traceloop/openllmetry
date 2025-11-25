from typing import Optional

from opentelemetry._logs import Logger
from opentelemetry.semconv_ai import SpanAttributes


class Config:
    exception_logger = None
    use_legacy_attributes = True
    event_logger: Optional[Logger] = None
    metadata_key_prefix: str = SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES
