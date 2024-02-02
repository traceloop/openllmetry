import json
import logging
from typing import Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.trace import SpanAttributes


logger = logging.getLogger(__name__)


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(SpanAttributes.DB_SYSTEM, "weaviate")
        span.set_attribute(SpanAttributes.DB_OPERATION, to_wrap.get("method"))

        obj = to_wrap.get("object")
        instrumentor = InstrumentorFactory.from_name(obj)
        if instrumentor:
            instrumentor.instrument(to_wrap.get("method"), span, args, kwargs)

        return_value = wrapped(*args, **kwargs)

    return return_value


def count_or_none(obj):
    if obj:
        return len(obj)

    return None


class ArgsGetter:
    """Helper to make sure we get arguments regardless
    of whether they were passed as args or as kwargs.

    Additionally, cast serializes dicts to JSON string.
    """

    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, index, name):
        try:
            obj = self.args[index]
        except IndexError:
            obj = self.kwargs.get(name)

        if obj:
            try:
                return json.dumps(obj)
            except json.decoder.JSONDecodeError:
                logger.warning(
                    "Failed to decode argument (%s) (%s) to JSON", index, name
                )


class _Instrumentor:
    def map_attributes(self, span, method_name, attributes, args, kwargs):
        getter = ArgsGetter(args, kwargs)
        for idx, attribute in enumerate(attributes):
            _set_span_attribute(
                span,
                f"{self.namespace}.{method_name}.{attribute}",
                getter(idx, attribute),
            )

    def instrument(self, method_name, span, args, kwargs):
        attributes = self.mapped_attributes.get(method_name)
        if attributes:
            self.map_attributes(span, method_name, attributes, args, kwargs)


class _SchemaInstrumentor(_Instrumentor):
    namespace = "db.weaviate.schema"
    mapped_attributes = {
        "get": ["class_name"],
        "create_class": ["schema_class"],
        "create": ["schema"],
        "delete_class": ["class_name"],
    }


class _DataObjectInstrumentor(_Instrumentor):
    namespace = "weaviate.data.crud_data"
    mapped_attributes = {
        "create": [
            "data_object",
            "class_name",
            "uuid",
            "vector",
            "consistency_level",
            "tenant",
        ],
        "validate": [
            "data_object",
            "class_name",
            "uuid",
            "vector",
        ],
        "get": [
            "uuid",
            "additional_properties",
            "with_vector",
            "class_name",
            "node_name",
            "consistency_level",
            "limit",
            "after",
            "offset",
            "sort",
            "tenant",
        ],
    }


class _BatchInstrumentor(_Instrumentor):
    namespace = "db.weaviate.batch"
    mapped_attributes = {
        "add_data_object": [
            "data_object",
            "class_name",
            "uuid",
            "vector",
            "tenant",
        ],
        "flush": [],
    }


class _QueryInstrumentor(_Instrumentor):
    namespace = "db.weaviate.query"
    mapped_attributes = {
        "get": [
            "class_name",
            "properties",
        ],
        "aggregate": ["class_name"],
        "raw": ["gql_query"],
    }


class _GetBuilderInstrumentor(_Instrumentor):
    namespace = "db.weaviate.query.get"
    mapped_attributes = {
        "do": [],
    }


class _GraphQLInstrumentor(_Instrumentor):
    namespace = "db.weaviate.query.filter"
    mapped_attributes = {
        "do": [],
    }


class InstrumentorFactory:
    @classmethod
    def from_name(cls, name: str) -> Optional[_Instrumentor]:
        if name == "Schema":
            return _SchemaInstrumentor()
        elif name == "DataObject":
            return _DataObjectInstrumentor()
        elif name == "Batch":
            return _BatchInstrumentor()
        elif name == "Query":
            return _QueryInstrumentor()
        elif name == "GetBuilder":
            return _GetBuilderInstrumentor()
        elif name == "GraphQL":
            return _GraphQLInstrumentor()
        return None
