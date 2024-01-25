"""OpenTelemetry Chroma DB instrumentation"""

import logging
import chromadb
from typing import Collection

from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.chromadb.wrapper import _wrap
from opentelemetry.instrumentation.chromadb.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("chromadb >= 0.3",)

WRAPPED_METHODS = [
    {
        "object": "Collection",
        "method": "add",
        "span_name": "chroma.add"
    },
    {
        "object": "Collection",
        "method": "get",
        "span_name": "chroma.get"
    },
    {
        "object": "Collection",
        "method": "peek",
        "span_name": "chroma.peek"
    },
    {
        "object": "Collection",
        "method": "query",
        "span_name": "chroma.query"
    },
    {
        "object": "Collection",
        "method": "modify",
        "span_name": "chroma.modify"
    },
    {
        "object": "Collection",
        "method": "update",
        "span_name": "chroma.update"
    },
    {
        "object": "Collection",
        "method": "upsert",
        "span_name": "chroma.upsert"
    },
    {
        "object": "Collection",
        "method": "delete",
        "span_name": "chroma.delete"
    }
]


class ChromaInstrumentor(BaseInstrumentor):
    """An instrumentor for Chroma's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            if getattr(chromadb, wrap_object, None):
                wrap_function_wrapper(
                    "chromadb",
                    f"{wrap_object}.{wrap_method}",
                    _wrap(tracer, wrapped_method),
                )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(f"chromadb.{wrap_object}", wrapped_method.get("method"))
