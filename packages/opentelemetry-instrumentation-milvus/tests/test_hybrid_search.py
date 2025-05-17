import os
import random
import pytest
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, DataType
from opentelemetry.semconv_ai import Events, SpanAttributes, EventAttributes

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "milvus.db")
client = MilvusClient(uri=path)


@pytest.fixture
def collection():
    collection_name = "my_hybrid_search"
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    # Add fields to schema
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=128)

    index_params = client.prepare_index_params()

    # Add indexes
    index_params.add_index(
        field_name="dense",
        index_name="dense_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )

    index_params.add_index(
        field_name="sparse",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
        metric_type="IP",  # Currently, only IP (Inner Product) is supported for sparse vectors
        params={
            "drop_ratio_build": 0.2
        },  # The ratio of small vector values to be dropped during indexing
    )

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    return collection_name


def insert_data(collection):

    data = [
        {
            "id": 0,
            "text": "Artificial intelligence was founded as an academic discipline in 1956.",
            "sparse": {9637: 0.30856525997853057, 4399: 0.19771651149001523},
            "dense": [random.random() for _ in range(128)],  # 128 dimensions
        },
        {
            "id": 1,
            "text": "Alan Turing was the first person to conduct substantial research in AI.",
            "sparse": {6959: 0.31025067641541815, 1729: 0.8265339135915016},
            "dense": [random.random() for _ in range(128)],  # 128 dimensions
        },
        {
            "id": 2,
            "text": "Born in Maida Vale, London, Turing was raised in southern England.",
            "sparse": {1220: 0.15303302147479103, 7335: 0.9436728846033107},
            "dense": [random.random() for _ in range(128)],  # 128 dimensions
        },
    ]

    client.insert(collection_name=collection, data=data)


def test_hybrid_search_with_rrf(exporter, collection):
    insert_data(collection)

    query_dense_vector = [random.random() for _ in range(128)]

    search_param_1 = {
        "data": [query_dense_vector],
        "anns_field": "dense",
        "param": {"metric_type": "IP", "params": {"nprobe": 10}},
        "limit": 10,
    }
    request_1 = AnnSearchRequest(**search_param_1)

    query_sparse_vector = {6959: 0.31025067641541815, 1729: 0.8265339135915016}
    search_param_2 = {
        "data": [query_sparse_vector],
        "anns_field": "sparse",
        "param": {"metric_type": "IP", "params": {"drop_ratio_build": 0.0}},
        "limit": 10,
    }
    request_2 = AnnSearchRequest(**search_param_2)

    reqs = [request_1, request_2]

    # RRF ranker
    ranker = RRFRanker(10)

    client.hybrid_search(collection_name=collection, reqs=reqs, ranker=ranker, limit=10)

    # Span checks
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.hybrid_search")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "hybrid_search"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_SEARCH_COLLECTION_NAME) == collection
    )
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_LIMIT) == 10

    reqs_info = []
    for req in reqs:
        req_info = {
            "anns_field": req.anns_field,
            "param": req.param,
        }
        reqs_info.append(req_info)

    assert span.attributes.get(
        SpanAttributes.MILVUS_SEARCH_ANNSEARCH_REQUEST
    ) == str(reqs_info)
    assert (
        span.attributes.get(SpanAttributes.MILVUS_SEARCH_RANKER_TYPE)
        == "RRFRanker"
    )

    # Result events
    events = [e for e in span.events if e.name == Events.DB_SEARCH_RESULT.value]

    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_RESULT_COUNT) == len(events)

    for event in events:
        _id = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_ID.value)
        score = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_DISTANCE.value)
        assert isinstance(_id, int)
        assert isinstance(float(score), float)
