"""OpenTelemetry Weaviate instrumentation"""

import logging
from weaviate import WeaviateClient

from typing import Collection

from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.weaviate.wrapper import _wrap
from opentelemetry.instrumentation.weaviate.version import __version__

print("HELLO WEAVIATE INIT")
logger = logging.getLogger(__name__)

_instruments = ("weaviate-client >= 4.4.0",)

WRAPPED_METHODS = [

    {
        "class": WeaviateClient,
        "object": "collections",
        "method": "get",
        "span_name": "weaviate.collections.get"
    },
]


class WeaviateInstrumentor(BaseInstrumentor):
    """An instrumentor for Weaviate's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            if getattr(wrap_package, wrap_object, None):
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(tracer, wrapped_method),
                )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")

            wrapped = getattr(wrap_package, wrap_object, None)
            if wrapped:
                unwrap(wrapped, wrapped_method.get("method"))
