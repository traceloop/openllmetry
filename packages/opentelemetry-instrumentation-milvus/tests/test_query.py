import os
import random

import pymilvus
import pytest
from opentelemetry.semconv_ai import Events, SpanAttributes

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "milvus.db")
milvus = pymilvus.MilvusClient(uri=path)


@pytest.fixture
def collection():
    collection_name = "Colors"
    milvus.create_collection(collection_name=collection_name, dimension=5)
    yield collection_name
    milvus.drop_collection(collection_name=collection_name)


def insert_data(collection):
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
    for i in data:
        i["color_tag"] = "{}_{}".format(i["color"], i["tag"])
    milvus.insert(collection_name=collection, data=data)


def test_milvus_insert(exporter, collection):
    insert_data(collection)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.insert")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "insert"
    assert span.attributes.get(SpanAttributes.MILVUS_INSERT_COLLECTION_NAME) == "Colors"
    assert span.attributes.get(SpanAttributes.MILVUS_INSERT_DATA_COUNT) == 1003


def test_milvus_query_equal(exporter, collection):
    insert_data(collection)
    milvus.query(
        collection_name=collection,
        filter='color == "brown"',
        output_fields=["color_tag"],
        limit=3,
    )
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.query")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "query"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_QUERY_COLLECTION_NAME) == collection
    )
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_FILTER) == 'color == "brown"'
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_OUTPUT_FIELDS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_LIMIT) == 3

    events = span.events
    for event in events:
        assert event.name == Events.DB_QUERY_RESULT.value
        tag = event.attributes.get("color_tag")
        _id = event.attributes.get("id")
        assert isinstance(tag, str)
        assert isinstance(_id, int)


def test_milvus_query_like(exporter, collection):
    insert_data(collection)
    milvus.query(
        collection_name=collection,
        filter='color_tag like "brown"',
        output_fields=["color_tag"],
        limit=2,
    )
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.query")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "query"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_QUERY_COLLECTION_NAME) == collection
    )
    assert (
        span.attributes.get(SpanAttributes.MILVUS_QUERY_FILTER)
        == 'color_tag like "brown"'
    )
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_OUTPUT_FIELDS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_LIMIT) == 2
    events = span.events
    for event in events:
        assert event.name == Events.DB_QUERY_RESULT.value
        tag = event.attributes.get("color_tag")
        _id = event.attributes.get("id")
        assert isinstance(tag, str)
        assert isinstance(_id, int)


def test_milvus_query_or(exporter, collection):
    insert_data(collection)
    milvus.query(
        collection_name=collection,
        filter='color == "brown" or color == "red"',
        output_fields=["color_tag"],
        limit=3,
    )
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.query")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "query"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_QUERY_COLLECTION_NAME) == collection
    )
    assert (
        span.attributes.get(SpanAttributes.MILVUS_QUERY_FILTER)
        == 'color == "brown" or color == "red"'
    )
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_OUTPUT_FIELDS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_LIMIT) == 3
    events = span.events
    for event in events:
        assert event.name == Events.DB_QUERY_RESULT.value
        tag = event.attributes.get("color_tag")
        _id = event.attributes.get("id")
        assert isinstance(tag, str)
        assert isinstance(_id, int)


def test_milvus_query_and(exporter, collection):
    insert_data(collection)
    milvus.query(
        collection_name=collection,
        filter='color == "brown" and tag == 1234',
        output_fields=["color_tag"],
        limit=1,
    )
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.query")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "query"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_QUERY_COLLECTION_NAME) == collection
    )
    assert (
        span.attributes.get(SpanAttributes.MILVUS_QUERY_FILTER)
        == 'color == "brown" and tag == 1234'
    )
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_OUTPUT_FIELDS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.MILVUS_QUERY_LIMIT) == 1
    events = span.events
    for event in events:
        assert event.name == Events.DB_QUERY_RESULT.value
        tag = event.attributes.get("color_tag")
        _id = event.attributes.get("id")
        assert isinstance(tag, str)
        assert isinstance(_id, int)
