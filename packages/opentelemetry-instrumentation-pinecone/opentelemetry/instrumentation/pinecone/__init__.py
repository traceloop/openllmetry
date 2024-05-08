"""OpenTelemetry Pinecone instrumentation"""

import logging
import json
from opentelemetry.instrumentation.pinecone.config import Config
from opentelemetry.instrumentation.pinecone.utils import dont_throw
import pinecone
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv.ai import EventAttributes, Events
from opentelemetry.instrumentation.pinecone.version import __version__

from opentelemetry.semconv.ai import SpanAttributes

logger = logging.getLogger(__name__)

_instruments = ("pinecone-client >= 2.2.2, <4",)


WRAPPED_METHODS = [
    {
        "object": "GRPCIndex",
        "method": "query",
        "span_name": "pinecone.query",
    },
    {
        "object": "GRPCIndex",
        "method": "upsert",
        "span_name": "pinecone.upsert",
    },
    {
        "object": "GRPCIndex",
        "method": "delete",
        "span_name": "pinecone.delete",
    },
    {
        "object": "Index",
        "method": "query",
        "span_name": "pinecone.query",
    },
    {
        "object": "Index",
        "method": "upsert",
        "span_name": "pinecone.upsert",
    },
    {
        "object": "Index",
        "method": "delete",
        "span_name": "pinecone.delete",
    },
]


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def _set_query_input_attributes(span, kwargs):
    # Pinecone-client 2.2.2 query kwargs
    # vector: Optional[List[float]] = None,
    # id: Optional[str] = None,
    # queries: Optional[Union[List[QueryVector], List[Tuple]]] = None,
    # top_k: Optional[int] = None,
    # namespace: Optional[str] = None,
    # filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
    # include_values: Optional[bool] = None,
    # include_metadata: Optional[bool] = None,
    # sparse_vector: Optional[Union[SparseValues, Dict[str, Union[List[float], List[int]]]]] = None,
    # **kwargs) -> QueryResponse:

    _set_span_attribute(span, "pinecone.query.id", kwargs.get("id"))
    _set_span_attribute(span, "pinecone.query.queries", kwargs.get("queries"))
    _set_span_attribute(span, "pinecone.query.top_k", kwargs.get("top_k"))
    _set_span_attribute(span, "pinecone.query.namespace", kwargs.get("namespace"))
    if isinstance(kwargs.get("filter"), dict):
        _set_span_attribute(
            span, "pinecone.query.filter", json.dumps(kwargs.get("filter"))
        )
    else:
        _set_span_attribute(span, "pinecone.query.filter", kwargs.get("filter"))
    _set_span_attribute(
        span, "pinecone.query.include_values", kwargs.get("include_values")
    )
    _set_span_attribute(
        span, "pinecone.query.include_metadata", kwargs.get("include_metadata")
    )

    # Log query embeddings
    # We assume user will pass either vector, sparse_vector or queries
    # But not two or more simultaneously
    # When defining conflicting sources of embeddings, the trace result is undefined

    vector = kwargs.get("vector")
    if vector:
        span.add_event(
            name="db.query.embeddings",
            attributes={"db.query.embeddings.vector": vector},
        )

    sparse_vector = kwargs.get("sparse_vector")
    if sparse_vector:
        span.add_event(
            name="db.query.embeddings",
            attributes={"db.query.embeddings.vector": sparse_vector},
        )

    queries = kwargs.get("queries")
    if queries:
        for vector in queries:
            span.add_event(
                name=Events.DB_QUERY_EMBEDDINGS.value,
                attributes={EventAttributes.DB_QUERY_EMBEDDINGS_VECTOR.value: vector},
            )


@dont_throw
def _set_query_response(span, response):
    matches = response.get("matches")

    for match in matches:
        span.add_event(
            name=Events.DB_QUERY_RESULT.value,
            attributes={
                EventAttributes.DB_QUERY_RESULT_ID.value: match.get("id"),
                EventAttributes.DB_QUERY_RESULT_SCORE.value: match.get("score"),
                EventAttributes.DB_QUERY_RESULT_METADATA.value: str(
                    match.get("metadata")
                ),
                EventAttributes.DB_QUERY_RESULT_VECTOR.value: match.get("values"),
            },
        )


def _set_input_attributes(span, kwargs):
    pass


def _set_response_attributes(span, response):
    if response.get("usage"):
        span.set_attribute(
            "pinecone.usage.read_units", response.get("usage").get("read_units") or 0
        )
        span.set_attribute(
            "pinecone.usage.write_units", response.get("usage").get("write_units") or 0
        )


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.VECTOR_DB_VENDOR: "Pinecone",
        },
    ) as span:
        if span.is_recording():
            if to_wrap.get("method") == "query":
                _set_query_input_attributes(span, kwargs)
            else:
                _set_input_attributes(span, kwargs)

        response = wrapped(*args, **kwargs)

        if response:
            if span.is_recording():
                if to_wrap.get("method") == "query":
                    _set_query_response(span, response)

                _set_response_attributes(span, response)

                span.set_status(Status(StatusCode.OK))

        return response


class PineconeInstrumentor(BaseInstrumentor):
    """An instrumentor for Pinecone's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            if getattr(pinecone, wrap_object, None):
                wrap_function_wrapper(
                    "pinecone",
                    f"{wrap_object}.{wrap_method}",
                    _wrap(tracer, wrapped_method),
                )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(f"pinecone.{wrap_object}", wrapped_method.get("method"))
