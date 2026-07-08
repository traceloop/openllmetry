from opentelemetry.instrumentation.chromadb.utils import dont_throw
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)
from opentelemetry.semconv_ai import EventAttributes, Events
from opentelemetry.semconv_ai import SpanAttributes as AISpanAttributes
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
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
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
            _set_segment_query_attributes(span, kwargs, args)
            _add_segment_query_embeddings_events(span, kwargs, args)
        elif to_wrap.get("method") == "modify":
            _set_modify_attributes(span, kwargs)
        elif to_wrap.get("method") == "update":
            _set_update_attributes(span, kwargs)
        elif to_wrap.get("method") == "upsert":
            _set_upsert_attributes(span, kwargs)
        elif to_wrap.get("method") == "delete":
            _set_delete_attributes(span, kwargs)

        try:
            return_value = wrapped(*args, **kwargs)
        except Exception as e:
            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
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


@dont_throw
def _set_add_attributes(span, kwargs):
    _set_span_attribute(
        span, AISpanAttributes.CHROMADB_ADD_IDS_COUNT, count_or_none(kwargs.get("ids"))
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_ADD_EMBEDDINGS_COUNT,
        count_or_none(kwargs.get("embeddings")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_ADD_METADATAS_COUNT,
        count_or_none(kwargs.get("metadatas")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_ADD_DOCUMENTS_COUNT,
        count_or_none(kwargs.get("documents")),
    )


@dont_throw
def _set_get_attributes(span, kwargs):
    _set_span_attribute(
        span, AISpanAttributes.CHROMADB_GET_IDS_COUNT, count_or_none(kwargs.get("ids"))
    )
    _set_span_attribute(
        span, AISpanAttributes.CHROMADB_GET_WHERE, _encode_where(kwargs.get("where"))
    )
    _set_span_attribute(span, AISpanAttributes.CHROMADB_GET_LIMIT, kwargs.get("limit"))
    _set_span_attribute(
        span, AISpanAttributes.CHROMADB_GET_OFFSET, kwargs.get("offset")
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_GET_WHERE_DOCUMENT,
        _encode_where_document(kwargs.get("where_document")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_GET_INCLUDE,
        _encode_include(kwargs.get("include")),
    )


@dont_throw
def _set_peek_attributes(span, kwargs):
    _set_span_attribute(span, AISpanAttributes.CHROMADB_PEEK_LIMIT, kwargs.get("limit"))


@dont_throw
def _set_query_attributes(span, kwargs):
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_QUERY_EMBEDDINGS_COUNT,
        count_or_none(kwargs.get("query_embeddings")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_QUERY_TEXTS_COUNT,
        count_or_none(kwargs.get("query_texts")),
    )
    _set_span_attribute(
        span, AISpanAttributes.CHROMADB_QUERY_N_RESULTS, kwargs.get("n_results")
    )
    _set_span_attribute(
        span, AISpanAttributes.CHROMADB_QUERY_WHERE, _encode_where(kwargs.get("where"))
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_QUERY_WHERE_DOCUMENT,
        _encode_where_document(kwargs.get("where_document")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_QUERY_INCLUDE,
        _encode_include(kwargs.get("include")),
    )


@dont_throw
def _set_segment_query_attributes(span, kwargs, args=None):
    # collection_id can be passed as positional arg[0] or keyword arg
    collection_id = kwargs.get("collection_id")
    if collection_id is None and args and len(args) > 0:
        collection_id = args[0]
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_QUERY_SEGMENT_QUERY_COLLECTION_ID,
        str(collection_id) if collection_id else None,
    )


@dont_throw
def _add_segment_query_embeddings_events(span, kwargs, args=None):
    # query_embeddings can be passed as positional arg[1] or keyword arg
    query_embeddings = kwargs.get("query_embeddings")
    if query_embeddings is None and args and len(args) > 1:
        query_embeddings = args[1]
    for embeddings in query_embeddings or []:
        span.add_event(
            name=Events.DB_QUERY_EMBEDDINGS.value,
            attributes={
                EventAttributes.DB_QUERY_EMBEDDINGS_VECTOR.value: json.dumps(
                    embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
                )
            },
        )


@dont_throw
def _add_query_result_events(span, kwargs):
    """
    ChromaDB query results have a nested structure — one inner list per query:

        {
            "ids":       [["id1", "id2", "id3"]],   <- outer: per query, inner: per result
            "distances": [[0.1,   0.2,   0.3  ]],
            "metadatas": [[{...}, {...}, {...} ]],
            "documents": [["doc1","doc2","doc3"]],
        }

    We emit one db.query.result span event per result document:

        event: db.query.result { id: "id1", distance: 0.1, document: "doc1", metadata: "..." }
        event: db.query.result { id: "id2", distance: 0.2, document: "doc2", metadata: "..." }
        event: db.query.result { id: "id3", distance: 0.3, document: "doc3", metadata: "..." }

    For N queries with n_results=K, this produces N×K events total.
    """
    # Outer zip: one tuple per query (ChromaDB returns a list-of-lists)
    for query_ids, query_distances, query_metadatas, query_documents in itertools.zip_longest(
        kwargs.get("ids", []) or [],
        kwargs.get("distances", []) or [],
        kwargs.get("metadatas", []) or [],
        kwargs.get("documents", []) or [],
    ):
        # Inner zip: one event per result document within this query
        for id_, distance, metadata, document in itertools.zip_longest(
            query_ids or [],
            query_distances or [],
            query_metadatas or [],
            query_documents or [],
        ):
            attributes = {}
            if id_ is not None:
                attributes[EventAttributes.DB_QUERY_RESULT_ID.value] = id_
            if distance is not None:
                attributes[EventAttributes.DB_QUERY_RESULT_DISTANCE.value] = distance
            if metadata is not None:
                attributes[EventAttributes.DB_QUERY_RESULT_METADATA.value] = (
                    json.dumps(metadata) if isinstance(metadata, dict) else metadata
                )
            if document is not None:
                attributes[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value] = document

            span.add_event(name=Events.DB_QUERY_RESULT.value, attributes=attributes)


@dont_throw
def _set_modify_attributes(span, kwargs):
    _set_span_attribute(span, AISpanAttributes.CHROMADB_MODIFY_NAME, kwargs.get("name"))
    # TODO: Add metadata attribute


@dont_throw
def _set_update_attributes(span, kwargs):
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_UPDATE_IDS_COUNT,
        count_or_none(kwargs.get("ids")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_UPDATE_EMBEDDINGS_COUNT,
        count_or_none(kwargs.get("embeddings")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_UPDATE_METADATAS_COUNT,
        count_or_none(kwargs.get("metadatas")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_UPDATE_DOCUMENTS_COUNT,
        count_or_none(kwargs.get("documents")),
    )


@dont_throw
def _set_upsert_attributes(span, kwargs):
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_UPSERT_EMBEDDINGS_COUNT,
        count_or_none(kwargs.get("embeddings")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_UPSERT_METADATAS_COUNT,
        count_or_none(kwargs.get("metadatas")),
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_UPSERT_DOCUMENTS_COUNT,
        count_or_none(kwargs.get("documents")),
    )


@dont_throw
def _set_delete_attributes(span, kwargs):
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_DELETE_IDS_COUNT,
        count_or_none(kwargs.get("ids")),
    )
    _set_span_attribute(
        span, AISpanAttributes.CHROMADB_DELETE_WHERE, _encode_where(kwargs.get("where"))
    )
    _set_span_attribute(
        span,
        AISpanAttributes.CHROMADB_DELETE_WHERE_DOCUMENT,
        _encode_where_document(kwargs.get("where_document")),
    )
