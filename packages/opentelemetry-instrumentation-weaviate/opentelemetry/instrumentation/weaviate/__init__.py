"""OpenTelemetry Weaviate instrumentation"""

import importlib
import logging
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.weaviate.version import __version__
from opentelemetry.instrumentation.weaviate.wrapper import _wrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper


logger = logging.getLogger(__name__)

_instruments = ("weaviate-client >= 3.26.0, <4",)


WRAPPED_METHODS = [
    {
        "module": "weaviate.schema",
        "object": "Schema",
        "method": "get",
        "span_name": "db.weaviate.schema.get",
    },
    {
        "module": "weaviate.schema",
        "object": "Schema",
        "method": "create_class",
        "span_name": "db.weaviate.schema.create_class",
    },
    {
        "module": "weaviate.schema",
        "object": "Schema",
        "method": "create",
        "span_name": "db.weaviate.schema.create",
    },
    {
        "module": "weaviate.schema",
        "object": "Schema",
        "method": "delete_class",
        "span_name": "db.weaviate.schema.delete_class",
    },
    {
        "module": "weaviate.schema",
        "object": "Schema",
        "method": "delete_all",
        "span_name": "db.weaviate.schema.delete_all",
    },
    {
        "module": "weaviate.data.crud_data",
        "object": "DataObject",
        "method": "create",
        "span_name": "db.weaviate.data.crud_data.create",
    },
    {
        "module": "weaviate.data.crud_data",
        "object": "DataObject",
        "method": "validate",
        "span_name": "db.weaviate.data.crud_data.validate",
    },
    {
        "module": "weaviate.data.crud_data",
        "object": "DataObject",
        "method": "get",
        "span_name": "db.weaviate.data.crud_data.get",
    },
    {
        "module": "weaviate.batch.crud_batch",
        "object": "Batch",
        "method": "add_data_object",
        "span_name": "db.weaviate.batch.crud_batch.add_data_object",
    },
    {
        "module": "weaviate.batch.crud_batch",
        "object": "Batch",
        "method": "flush",
        "span_name": "db.weaviate.batch.crud_batch.flush",
    },
    {
        "module": "weaviate.batch.crud_batch",
        "object": "Batch",
        "method": "flush",
        "span_name": "db.weaviate.batch.crud_batch.flush",
    },
    {
        "module": "weaviate.gql.query",
        "object": "Query",
        "method": "get",
        "span_name": "db.weaviate.gql.query.get",
    },
    {
        "module": "weaviate.gql.query",
        "object": "Query",
        "method": "aggregate",
        "span_name": "db.weaviate.gql.query.aggregate",
    },
    {
        "module": "weaviate.gql.query",
        "object": "Query",
        "method": "raw",
        "span_name": "db.weaviate.gql.query.raw",
    },
    {
        "module": "weaviate.gql.get",
        "object": "GetBuilder",
        "method": "do",
        "span_name": "db.weaviate.gql.query.get.do",
    },
    {
        "module": "weaviate.gql.filter",
        "object": "GraphQL",
        "method": "do",
        "span_name": "db.weaviate.gql.query.filter.do",
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
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            module = importlib.import_module(wrap_module)
            if getattr(module, wrap_object, None):
                wrap_function_wrapper(
                    wrap_module,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(tracer, wrapped_method),
                )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            module = importlib.import_module(wrap_module)
            wrapped = getattr(module, wrap_object, None)
            if wrapped:
                unwrap(wrapped, wrapped_method.get("method"))
