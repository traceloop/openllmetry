"""Tests for AsyncMilvusClient instrumentation (pymilvus>=2.6.0; async _async_with_tracer path)."""

import os
import random

import pytest
import pytest_asyncio
from opentelemetry.semconv_ai import Events, SpanAttributes, EventAttributes, Meters
from opentelemetry.trace.status import StatusCode
from pymilvus import AnnSearchRequest, AsyncMilvusClient, DataType, MilvusClient, RRFRanker

from .utils import find_metrics_by_name

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "milvus.db")


@pytest_asyncio.fixture
async def client():
    c = AsyncMilvusClient(uri=path)
    yield c


@pytest_asyncio.fixture
async def collection(client):
    collection_name = "Colors"
    await client.create_collection(collection_name=collection_name, dimension=5)
    yield collection_name
    await client.drop_collection(collection_name=collection_name)


@pytest_asyncio.fixture
async def hybrid_collection(client):
    collection_name = "my_hybrid_search"
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=128)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="dense",
        index_name="dense_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )

    index_params.add_index(
        field_name="sparse",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2},
    )

    await client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    yield collection_name
    await client.drop_collection(collection_name=collection_name)


async def insert_data(client: AsyncMilvusClient, collection: str) -> None:
    colors = [
        "green",
        "blue",
        "yellow",
        "red",
        "black",
        "white",
        "purple",
        "pink",
        "orange",
        "grey",
    ]
    data = [
        {
            "id": i,
            "vector": [random.uniform(-1, 1) for _ in range(5)],
            "color": random.choice(colors),
            "tag": random.randint(1000, 9999),
        }
        for i in range(1000)
    ]
    data += [
        {
            "id": 1000,
            "vector": [random.uniform(-1, 1) for _ in range(5)],
            "color": "brown",
            "tag": 1234,
        },
        {
            "id": 1001,
            "vector": [random.uniform(-1, 1) for _ in range(5)],
            "color": "brown",
            "tag": 5678,
        },
        {
            "id": 1002,
            "vector": [random.uniform(-1, 1) for _ in range(5)],
            "color": "brown",
            "tag": 9101,
        },
    ]
    for row in data:
        row["color_tag"] = "{}_{}".format(row["color"], row["tag"])
    await client.insert(collection_name=collection, data=data)


async def test_async_milvus_single_vector_search(exporter, collection, client, reader):
    await insert_data(client, collection)

    query_vectors = [[random.uniform(-1, 1) for _ in range(5)]]
    search_params = {"radius": 0.5, "metric_type": "COSINE", "index_type": "IVF_FLAT"}
    await client.search(
        collection_name=collection,
        data=query_vectors,
        anns_field="vector",
        search_params=search_params,
        output_fields=["color_tag"],
        limit=3,
        timeout=10,
    )

    spans = exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "milvus.search")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "search"
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_COLLECTION_NAME) == collection
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_OUTPUT_FIELDS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_LIMIT) == 3
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_TIMEOUT) == 10
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_ANNS_FIELD) == "vector"
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_QUERY_VECTOR_DIMENSION) == "[5]"

    events = span.events
    for event in events:
        assert event.name == Events.DB_SEARCH_RESULT.value
        _id = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_ID.value)
        distance = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_DISTANCE.value)
        assert isinstance(_id, int)
        assert isinstance(distance, str)

    total_matches = len(events)
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_RESULT_COUNT) == total_matches

    metrics_data = reader.get_metrics_data()
    distance_metrics = find_metrics_by_name(metrics_data, Meters.DB_SEARCH_DISTANCE)
    for metric in distance_metrics:
        assert all(dp.sum >= 0 for dp in metric.data.data_points)


async def test_async_milvus_query(exporter, collection, client, reader):
    await insert_data(client, collection)

    await client.query(
        collection_name=collection,
        filter='color == "brown"',
        output_fields=["color_tag"],
        limit=3,
    )

    spans = exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "milvus.query")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "query"
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_COLLECTION_NAME) == collection
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_FILTER) == 'color == "brown"'
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_OUTPUT_FIELDS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_LIMIT) == 3

    metrics_data = reader.get_metrics_data()
    duration_metrics = find_metrics_by_name(metrics_data, Meters.DB_QUERY_DURATION)
    for metric in duration_metrics:
        assert all(dp.sum >= 0 for dp in metric.data.data_points)

    for event in span.events:
        assert event.name == Events.DB_QUERY_RESULT.value
        tag = event.attributes.get("color_tag")
        _id = event.attributes.get("id")
        assert isinstance(tag, str)
        assert isinstance(_id, int)


async def test_async_milvus_insert_upsert_get_delete(exporter, collection, client, reader):
    await insert_data(client, collection)

    spans = exporter.get_finished_spans()
    insert_span = next(s for s in spans if s.name == "milvus.insert")
    assert insert_span.attributes.get(SpanAttributes.MILVUS_INSERT_DATA_COUNT) == 1003

    modified = {
        "id": 1000,
        "vector": [random.uniform(-1, 1) for _ in range(5)],
        "color": "red",
        "tag": 1234,
    }
    await client.upsert(collection_name=collection, data=modified)

    spans = exporter.get_finished_spans()
    upsert_span = next(s for s in reversed(spans) if s.name == "milvus.upsert")
    assert upsert_span.attributes.get(SpanAttributes.MILVUS_UPSERT_COLLECTION_NAME) == collection

    await client.get(
        collection_name=collection,
        ids=[1000, 1001, 1002],
        output_fields=["color_tag"],
        timeout=10,
    )

    spans = exporter.get_finished_spans()
    get_span = next(s for s in reversed(spans) if s.name == "milvus.get")
    assert get_span.attributes.get(SpanAttributes.MILVUS_GET_IDS_COUNT) == 3

    await client.delete(collection_name=collection, ids=[1000, 1001, 1002], timeout=10)

    spans = exporter.get_finished_spans()
    delete_span = next(s for s in reversed(spans) if s.name == "milvus.delete")
    assert delete_span.attributes.get(SpanAttributes.MILVUS_DELETE_IDS_COUNT) == 3

    metrics_data = reader.get_metrics_data()
    insert_metrics = find_metrics_by_name(metrics_data, Meters.DB_USAGE_INSERT_UNITS)
    for metric in insert_metrics:
        assert all(dp.value == 1003 for dp in metric.data.data_points)
    upsert_metrics = find_metrics_by_name(metrics_data, Meters.DB_USAGE_UPSERT_UNITS)
    for metric in upsert_metrics:
        assert all(dp.value == 1 for dp in metric.data.data_points)
    delete_metrics = find_metrics_by_name(metrics_data, Meters.DB_USAGE_DELETE_UNITS)
    for metric in delete_metrics:
        assert all(dp.value == 3 for dp in metric.data.data_points)


async def test_async_milvus_search_error(exporter, collection, client):
    await insert_data(client, collection)

    query_vector = [random.uniform(-1, 1) for _ in range(5)]
    search_params = {"radius": 0.5, "metric_type": "COSINE", "index_type": "IVF_FLAT"}

    with pytest.raises(Exception):
        await client.search(
            collection_name="nonexistent_async_collection",
            data=[query_vector],
            anns_field="vector",
            search_params=search_params,
            output_fields=["color_tag"],
            limit=3,
            timeout=10,
        )

    spans = exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "milvus.search")
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "COLLECTION_NOT_FOUND"


async def test_async_milvus_hybrid_search(exporter, client, hybrid_collection):
    data = [
        {
            "id": 0,
            "text": "Artificial intelligence was founded as an academic discipline in 1956.",
            "sparse": {9637: 0.30856525997853057, 4399: 0.19771651149001523},
            "dense": [random.random() for _ in range(128)],
        },
        {
            "id": 1,
            "text": "Alan Turing was the first person to conduct substantial research in AI.",
            "sparse": {6959: 0.31025067641541815, 1729: 0.8265339135915016},
            "dense": [random.random() for _ in range(128)],
        },
        {
            "id": 2,
            "text": "Born in Maida Vale, London, Turing was raised in southern England.",
            "sparse": {1220: 0.15303302147479103, 7335: 0.9436728846033107},
            "dense": [random.random() for _ in range(128)],
        },
    ]

    await client.insert(collection_name=hybrid_collection, data=data)
    query_dense = [random.random() for _ in range(128)]
    request_1 = AnnSearchRequest(
        data=[query_dense],
        anns_field="dense",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=10,
    )
    query_sparse = {6959: 0.31025067641541815, 1729: 0.8265339135915016}
    request_2 = AnnSearchRequest(
        data=[query_sparse],
        anns_field="sparse",
        param={"metric_type": "IP", "params": {"drop_ratio_build": 0.0}},
        limit=10,
    )
    reqs = [request_1, request_2]
    ranker = RRFRanker(10)
    await client.hybrid_search(collection_name=hybrid_collection, reqs=reqs, ranker=ranker, limit=10)

    spans = exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "milvus.hybrid_search")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "hybrid_search"
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_COLLECTION_NAME) == hybrid_collection
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_LIMIT) == 10

    reqs_info = [{"anns_field": req.anns_field, "param": req.param} for req in reqs]
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_ANNSEARCH_REQUEST) == str(reqs_info)
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_RANKER_TYPE) == "RRFRanker"

    events = [e for e in span.events if e.name == Events.DB_SEARCH_RESULT.value]
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_RESULT_COUNT) == len(events)
    for event in events:
        _id = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_ID.value)
        score = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_DISTANCE.value)
        assert isinstance(_id, int)
        assert isinstance(float(score), float)
