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
    sparse_vector = kwargs.get("sparse_vector")
    queries = kwargs.get("queries")

    # Count of query embeddings: 1 for vector/sparse_vector, len() for queries.
    # Omit attribute when caller used id-only lookup (none of the three supplied).
    embeddings_count = None
    if queries is not None:
        embeddings_count = len(queries)
    elif vector is not None:
        embeddings_count = 1
    elif sparse_vector is not None:
        embeddings_count = 1

    if embeddings_count is not None:
        span.set_attribute(
            SpanAttributes.PINECONE_QUERY_EMBEDDINGS_COUNT, embeddings_count
        )
        span.set_attribute(
            SpanAttributes.VECTOR_DB_QUERY_EMBEDDINGS_COUNT, embeddings_count
        )

    if vector:
        span.add_event(
            name=f"{Events.DB_QUERY_EMBEDDINGS.value}",
            attributes={f"{EventAttributes.DB_QUERY_EMBEDDINGS_VECTOR.value}": vector},
        )

    if sparse_vector:
        span.add_event(
            name=f"{Events.DB_QUERY_EMBEDDINGS.value}",
            attributes={
                f"{EventAttributes.DB_QUERY_EMBEDDINGS_VECTOR.value}": sparse_vector
            },
        )

    if queries:
        for vector in queries:
            span.add_event(
                name=Events.DB_QUERY_EMBEDDINGS.value,
                attributes={EventAttributes.DB_QUERY_EMBEDDINGS_VECTOR.value: vector},
            )


@dont_throw
def set_query_response(span, scores_metric, shared_attributes, response):
    matches = response.get("matches") or []

    # result_count: always emit on a successful response (0 is meaningful — OTel spec).
    span.set_attribute(SpanAttributes.PINECONE_QUERY_RESULT_COUNT, len(matches))
    span.set_attribute(SpanAttributes.VECTOR_DB_QUERY_RESULT_COUNT, len(matches))

    scores = [m.get("score") for m in matches if m.get("score") is not None]
    if scores:
        span.set_attribute(SpanAttributes.VECTOR_DB_QUERY_TOP_SCORE, max(scores))

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
