"""OpenTelemetry Writer instrumentation"""
import logging
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.writer.config import Config
from opentelemetry.instrumentation.utils import (
    unwrap,
)

logger = logging.getLogger(__name__)

_instruments = ("writer-sdk >= 2.2.1, < 3",)

WRAPPED_METHODS = [
    {
        "method": "create",
        "span_name": "writer.completions",
    },
    {
        "method": "chat",
        "span_name": "writer.chat",
    },
]


class WriterInstrumentor(BaseInstrumentor):
    """An instrumentor for Writer's client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes=True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        ...

    def _uninstrument(self, **kwargs):
        try:
            import writerai
            from writerai import Writer, AsyncWriter

            for wrapped_method in WRAPPED_METHODS:
                method_name = wrapped_method.get("method")
                unwrap(Writer, method_name)
                unwrap(AsyncWriter, method_name)
                unwrap(writerai, method_name)
        except ImportError:
            logger.warning("Failed to import writer modules for uninstrumentation.")
