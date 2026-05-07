import json

from opentelemetry.semconv_ai import EventAttributes, Events, SpanAttributes
from opentelemetry.instrumentation.pinecone.utils import dont_throw, set_span_attribute


@dont_throw
def set_query_input_attributes(span, kwargs):
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

    set_span_attribute(span, SpanAttributes.PINECONE_QUERY_ID, kwargs.get("id"))
    set_span_attribute(
        span, SpanAttributes.PINECONE_QUERY_QUERIES, kwargs.get("queries")
    )
    set_span_attribute(span, SpanAttributes.PINECONE_QUERY_TOP_K, kwargs.get("top_k"))
    set_span_attribute(
        span, SpanAttributes.PINECONE_QUERY_NAMESPACE, kwargs.get("namespace")
    )
    if isinstance(kwargs.get("filter"), dict):
        set_span_attribute(
            span, SpanAttributes.PINECONE_QUERY_FILTER, json.dumps(kwargs.get("filter"))
        )
    else:
        set_span_attribute(
            span, SpanAttributes.PINECONE_QUERY_FILTER, kwargs.get("filter")
        )
    set_span_attribute(
        span, SpanAttributes.PINECONE_QUERY_INCLUDE_VALUES, kwargs.get("include_values")
    )
    set_span_attribute(
        span,
        SpanAttributes.PINECONE_QUERY_INCLUDE_METADATA,
        kwargs.get("include_metadata"),
    )

    # Log query embeddings
    # We assume user will pass either vector, sparse_vector or queries
    # But not two or more simultaneously
    # When defining conflicting sources of embeddings, the trace result is undefined

    vector = kwargs.get("vector")
    if vector:
        span.add_event(
            name=f"{Events.DB_QUERY_EMBEDDINGS.value}",
            attributes={f"{EventAttributes.DB_QUERY_EMBEDDINGS_VECTOR.value}": vector},
        )

    sparse_vector = kwargs.get("sparse_vector")
    if sparse_vector:
        span.add_event(
            name=f"{Events.DB_QUERY_EMBEDDINGS.value}",
            attributes={
                f"{EventAttributes.DB_QUERY_EMBEDDINGS_VECTOR.value}": sparse_vector
            },
        )

    queries = kwargs.get("queries")
    if queries:
        for vector in queries:
            span.add_event(
                name=Events.DB_QUERY_EMBEDDINGS.value,
                attributes={EventAttributes.DB_QUERY_EMBEDDINGS_VECTOR.value: vector},
            )


@dont_throw
def set_query_response(span, scores_metric, shared_attributes, response):
    matches = response.get("matches")

    for match in matches:
        if scores_metric and match.get("score"):
            scores_metric.record(match.get("score"), shared_attributes)

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


@dont_throw
def set_upsert_input_attributes(span, kwargs, args):
    # Pinecone upsert kwargs (positional or keyword):
    # vectors: List[Vector],
    # namespace: Optional[str] = None,
    # batch_size: Optional[int] = None,
    # async_req: bool = False,
    # show_progress: bool = True,

    vectors = kwargs.get("vectors")
    if vectors is None and args:
        vectors = args[0]
    if vectors is not None:
        try:
            set_span_attribute(
                span, SpanAttributes.PINECONE_UPSERT_VECTORS_COUNT, len(vectors)
            )
        except TypeError:
            # vectors is a generator or other non-len-supporting iterable;
            # don't materialise it just to count.
            pass
    set_span_attribute(
        span, SpanAttributes.PINECONE_UPSERT_NAMESPACE, kwargs.get("namespace")
    )
    set_span_attribute(
        span, SpanAttributes.PINECONE_UPSERT_BATCH_SIZE, kwargs.get("batch_size")
    )


@dont_throw
def set_delete_input_attributes(span, kwargs, args):
    # Pinecone delete kwargs (positional or keyword):
    # ids: Optional[List[str]] = None,
    # delete_all: Optional[bool] = None,
    # namespace: Optional[str] = None,
    # filter: Optional[Dict] = None,

    ids = kwargs.get("ids")
    if ids is None and args:
        ids = args[0]
    if ids is not None:
        try:
            set_span_attribute(
                span, SpanAttributes.PINECONE_DELETE_IDS_COUNT, len(ids)
            )
        except TypeError:
            pass
    set_span_attribute(
        span, SpanAttributes.PINECONE_DELETE_DELETE_ALL, kwargs.get("delete_all")
    )
    set_span_attribute(
        span, SpanAttributes.PINECONE_DELETE_NAMESPACE, kwargs.get("namespace")
    )
    if isinstance(kwargs.get("filter"), dict):
        set_span_attribute(
            span,
            SpanAttributes.PINECONE_DELETE_FILTER,
            json.dumps(kwargs.get("filter")),
        )
    else:
        set_span_attribute(
            span, SpanAttributes.PINECONE_DELETE_FILTER, kwargs.get("filter")
        )
