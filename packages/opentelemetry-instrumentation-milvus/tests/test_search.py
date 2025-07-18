import os
import random

import pymilvus
import pytest
from opentelemetry.semconv_ai import Events, SpanAttributes, EventAttributes, Meters
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


def test_milvus_single_vector_search(exporter, collection, reader):
    insert_data(collection)

    query_vectors = [
        [random.uniform(-1, 1) for _ in range(5)],  # Random query vector for the search
    ]
    search_params = {"radius": 0.5, "metric_type": "COSINE", "index_type": "IVF_FLAT"}
    milvus.search(
        collection_name=collection,
        data=query_vectors,
        anns_field="vector",
        search_params=search_params,
        output_fields=["color_tag"],
        limit=3,
        timeout=10,
    )

    # Get finished spans
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.search")

    # Check the span attributes related to search
    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "search"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_SEARCH_COLLECTION_NAME) == collection
    )
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_OUTPUT_FIELDS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_LIMIT) == 3
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_TIMEOUT) == 10
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_ANNS_FIELD) == "vector"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_SEARCH_QUERY_VECTOR_DIMENSION)
        == "[5]"
    )
    distances = []
    ids = []

    events = span.events

    for event in events:
        assert event.name == Events.DB_SEARCH_RESULT.value
        _id = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_ID.value)
        distance = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_DISTANCE.value)

        assert isinstance(_id, int)
        assert isinstance(distance, str)

        # Collect the distances and IDs for further computation
        distances.append(
            float(distance)
        )  # Convert the distance to a float for computation
        ids.append(_id)

    # Now compute dynamic stats from the distances
    total_matches = len(events)

    assert (
        span.attributes.get(SpanAttributes.MILVUS_SEARCH_RESULT_COUNT) == total_matches
    )
    metrics_data = reader.get_metrics_data()
    distance_metrics = find_metrics_by_name(metrics_data, Meters.DB_SEARCH_DISTANCE)
    for metric in distance_metrics:
        assert all(dp.sum >= 0 for dp in metric.data.data_points)


def test_milvus_multiple_vector_search(exporter, collection):
    insert_data(collection)

    query_vectors = [
        [random.uniform(-1, 1) for _ in range(5)],  # Random query vector for the search
        [random.uniform(-1, 1) for _ in range(5)],  # Another query vector
        [
            random.uniform(-1, 1) for _ in range(5)
        ],  # Another query vector (you can add more as needed)
    ]
    search_params = {"radius": 0.5, "metric_type": "COSINE", "index_type": "IVF_FLAT"}
    milvus.search(
        collection_name=collection,
        data=query_vectors,
        anns_field="vector",
        search_params=search_params,
        output_fields=["color_tag"],
        limit=3,
        timeout=10,
    )

    # Get finished spans
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.search")

    # Check the span attributes related to search
    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "milvus"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "search"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_SEARCH_COLLECTION_NAME) == collection
    )
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_OUTPUT_FIELDS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_LIMIT) == 3
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_TIMEOUT) == 10
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_ANNS_FIELD) == "vector"
    assert (
        span.attributes.get(SpanAttributes.MILVUS_SEARCH_QUERY_VECTOR_DIMENSION)
        == "[5, 5, 5]"
    )

    distances_dict = {}
    ids_dict = {}

    events = span.events
    for event in events:
        assert event.name == Events.DB_SEARCH_RESULT.value
        query_idx = event.attributes.get(
            EventAttributes.DB_SEARCH_RESULT_QUERY_ID.value
        )
        _id = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_ID.value)
        distance = event.attributes.get(EventAttributes.DB_SEARCH_RESULT_DISTANCE.value)

        assert isinstance(_id, int)
        assert isinstance(distance, str)

        distance = float(distance)

        if query_idx not in distances_dict:
            distances_dict[query_idx] = []
            ids_dict[query_idx] = []

        distances_dict[query_idx].append(distance)
        ids_dict[query_idx].append(_id)

    for query_idx in distances_dict:
        distances = distances_dict[query_idx]

        total_matches = len(distances)

        count_key = f"{SpanAttributes.MILVUS_SEARCH_RESULT_COUNT}_{query_idx}"

        assert span.attributes.get(count_key) == total_matches
