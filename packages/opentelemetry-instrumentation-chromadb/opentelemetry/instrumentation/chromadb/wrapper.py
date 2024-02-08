from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv.ai import Events, EventAttributes

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)
import itertools
import json


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
        span.set_attribute(SpanAttributes.DB_SYSTEM, "chroma")
        span.set_attribute(SpanAttributes.DB_OPERATION, to_wrap.get("method"))

        if to_wrap.get("method") == "add":
            _set_add_attributes(span, kwargs)
        elif to_wrap.get("method") == "get":
            _set_get_attributes(span, kwargs)
        elif to_wrap.get("method") == "peek":
            _set_peek_attributes(span, kwargs)
        elif to_wrap.get("method") == "query":
            _set_query_attributes(span, kwargs)
        elif to_wrap.get("method") == "_query":
            _set_segment_query_attributes(span, kwargs)
            _add_segment_query_embeddings_events(span, kwargs)
        elif to_wrap.get("method") == "modify":
            _set_modify_attributes(span, kwargs)
        elif to_wrap.get("method") == "update":
            _set_update_attributes(span, kwargs)
        elif to_wrap.get("method") == "upsert":
            _set_upsert_attributes(span, kwargs)
        elif to_wrap.get("method") == "delete":
            _set_delete_attributes(span, kwargs)

        return_value = wrapped(*args, **kwargs)
        if to_wrap.get("method") == "query":
            _add_query_result_events(span, return_value)

    return return_value


def _encode_where(where):
    where_str = None
    if where:
        where_str = str(where)

    return where_str


def _encode_where_document(where_document):
    where_document_str = None
    if where_document:
        where_document_str = str(where_document)

    return where_document_str


def _encode_include(include):
    include_str = None
    if include:
        include_str = str(include)

    return include_str


def count_or_none(obj):
    if obj:
        return len(obj)

    return None


def _set_add_attributes(span, kwargs):
    _set_span_attribute(span, "db.chroma.add.ids_count", count_or_none(kwargs.get("ids")))
    _set_span_attribute(span, "db.chroma.add.embeddings_count", count_or_none(kwargs.get("embeddings")))
    _set_span_attribute(span, "db.chroma.add.metadatas_count", count_or_none(kwargs.get("metadatas")))
    _set_span_attribute(span, "db.chroma.add.documents_count", count_or_none(kwargs.get("documents")))


def _set_get_attributes(span, kwargs):
    _set_span_attribute(span, "db.chroma.get.ids_count", count_or_none(kwargs.get("ids")))
    _set_span_attribute(span, "db.chroma.get.where", _encode_where(kwargs.get("where")))
    _set_span_attribute(span, "db.chroma.get.limit", kwargs.get("limit"))
    _set_span_attribute(span, "db.chroma.get.offset", kwargs.get("offset"))
    _set_span_attribute(span, "db.chroma.get.where_document", _encode_where_document(kwargs.get("where_document")))
    _set_span_attribute(span, "db.chroma.get.include", _encode_include(kwargs.get("include")))


def _set_peek_attributes(span, kwargs):
    _set_span_attribute(span, "db.chroma.peek.limit", kwargs.get("limit"))


def _set_query_attributes(span, kwargs):
    _set_span_attribute(span, "db.chroma.query.query_embeddings_count", count_or_none(kwargs.get("query_embeddings")))
    _set_span_attribute(span, "db.chroma.query.query_texts_count", count_or_none(kwargs.get("query_texts")))
    _set_span_attribute(span, "db.chroma.query.n_results", kwargs.get("n_results"))
    _set_span_attribute(span, "db.chroma.query.where", _encode_where(kwargs.get("where")))
    _set_span_attribute(span, "db.chroma.query.where_document", _encode_where_document(kwargs.get("where_document")))
    _set_span_attribute(span, "db.chroma.query.include", _encode_include(kwargs.get("include")))


def _set_segment_query_attributes(span, kwargs):
    _set_span_attribute(span, "db.chroma.query.segment._query.collection_id", str(kwargs.get("collection_id")))


def _add_segment_query_embeddings_events(span, kwargs):
    for i, embeddings in enumerate(kwargs.get("query_embeddings", [])):
        span.add_event(
            name=f"{Events.VECTOR_DB_QUERY_EMBEDDINGS.value}.{i}",
            attributes={
                f"{Events.VECTOR_DB_QUERY_EMBEDDINGS.value}.{i}.vector": json.dumps(embeddings)
            }
        )


def _add_query_result_events(span, kwargs):
    zipped = itertools.zip_longest(
        kwargs.get("ids", []),
        kwargs.get("distances", []),
        kwargs.get("metadata", []),
        kwargs.get("documents", [])
    )
    for i, tuple_ in enumerate(zipped):
        span.add_event(
            name=f"{Events.VECTOR_DB_QUERY_RESULT.value}.{i}",
            attributes={
                f"{EventAttributes.VECTOR_DB_QUERY_RESULT_IDS.value.format(i=i)}": tuple_[0] or [],
                f"{EventAttributes.VECTOR_DB_QUERY_RESULT_DISTANCES.value.format(i=i)}": tuple_[1] or [],
                f"{EventAttributes.VECTOR_DB_QUERY_RESULT_METADATA.value.format(i=i)}": tuple_[2] or [],
                f"{EventAttributes.VECTOR_DB_QUERY_RESULT_DOCUMENTS.value.format(i=i)}": tuple_[3] or [],
            }
        )


def _set_modify_attributes(span, kwargs):
    _set_span_attribute(span, "db.chroma.modify.name", kwargs.get("name"))
    # TODO: Add metadata attribute


def _set_update_attributes(span, kwargs):
    _set_span_attribute(span, "db.chroma.update.ids_count", count_or_none(kwargs.get("ids")))
    _set_span_attribute(span, "db.chroma.update.embeddings_count", count_or_none(kwargs.get("embeddings")))
    _set_span_attribute(span, "db.chroma.update.metadatas_count", count_or_none(kwargs.get("metadatas")))
    _set_span_attribute(span, "db.chroma.update.documents_count", count_or_none(kwargs.get("documents")))


def _set_upsert_attributes(span, kwargs):
    _set_span_attribute(span, "db.chroma.upsert.embeddings_count", count_or_none(kwargs.get("embeddings")))
    _set_span_attribute(span, "db.chroma.upsert.metadatas_count", count_or_none(kwargs.get("metadatas")))
    _set_span_attribute(span, "db.chroma.upsert.documents_count", count_or_none(kwargs.get("documents")))


def _set_delete_attributes(span, kwargs):
    _set_span_attribute(span, "db.chroma.delete.ids_count", count_or_none(kwargs.get("ids")))
    _set_span_attribute(span, "db.chroma.delete.where", _encode_where(kwargs.get("where")))
    _set_span_attribute(span, "db.chroma.delete.where_document", _encode_where_document(kwargs.get("where_document")))
