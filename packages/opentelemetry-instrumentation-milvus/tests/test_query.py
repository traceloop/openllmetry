import os
import random

import pymilvus
import pytest
from opentelemetry.semconv_ai import Events, SpanAttributes, Meters
from .utils import find_metrics_by_name

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


def test_milvus_insert(exporter, collection, reader):
    insert_data(collection)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.insert")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "insert"
    assert span.attributes.get(SpanAttributes.MILVUS_INSERT_COLLECTION_NAME) == "Colors"
    assert span.attributes.get(SpanAttributes.MILVUS_INSERT_DATA_COUNT) == 1003

    metrics_data = reader.get_metrics_data()
    insert_metrics = find_metrics_by_name(metrics_data, Meters.DB_USAGE_INSERT_UNITS)
    for metric in insert_metrics:
        assert all(dp.value == 1003 for dp in metric.data.data_points)


def test_milvus_upsert(exporter, collection, reader):
    insert_data(collection)
    modified_data = {
            "id": 1000,
            "vector": [random.uniform(-1, 1) for _ in range(5)],
            "color": "red",
            "tag": 1234,
        }
    milvus.upsert(collection_name=collection, data=modified_data)
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.upsert")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "upsert"
    assert span.attributes.get(SpanAttributes.MILVUS_UPSERT_COLLECTION_NAME) == "Colors"

    metrics_data = reader.get_metrics_data()
    upsert_metrics = find_metrics_by_name(metrics_data, Meters.DB_USAGE_UPSERT_UNITS)
    for metric in upsert_metrics:
        assert all(dp.value == 1 for dp in metric.data.data_points)


def test_milvus_query_equal(exporter, collection, reader):
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
    metrics_data = reader.get_metrics_data()
    duration_metrics = find_metrics_by_name(metrics_data, Meters.DB_QUERY_DURATION)
    for metric in duration_metrics:
        assert all(dp.sum >= 0 for dp in metric.data.data_points)

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


def test_milvus_get_collection(exporter, collection):
    insert_data(collection)
    milvus.get(
        collection_name=collection,
        ids=[1000, 1001, 1002],
        output_fields=["color_tag"],
        timeout=10,
    )
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.get")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "get"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_GET_COLLECTION_NAME) == collection
    )
    assert span.attributes.get(SpanAttributes.MILVUS_GET_OUTPUT_FIELDS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.MILVUS_GET_TIMEOUT) == 10
    assert span.attributes.get(SpanAttributes.MILVUS_GET_IDS_COUNT) == 3


def test_milvus_delete_collection(exporter, collection, reader):
    insert_data(collection)
    milvus.delete(
        collection_name=collection, ids=[1000, 1001, 1002], timeout=10
    )
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.delete")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "delete"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_DELETE_COLLECTION_NAME) == collection
    )
    assert span.attributes.get(SpanAttributes.MILVUS_DELETE_IDS_COUNT) == 3
    assert span.attributes.get(SpanAttributes.MILVUS_DELETE_TIMEOUT) == 10
    metrics_data = reader.get_metrics_data()
    delete_metrics = find_metrics_by_name(
        metrics_data, Meters.DB_USAGE_DELETE_UNITS
    )
    for metric in delete_metrics:
        assert all(dp.value == 3 for dp in metric.data.data_points)
