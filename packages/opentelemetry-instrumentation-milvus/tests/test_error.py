import os
import random

import pymilvus
import pytest
from opentelemetry.semconv_ai import Events, SpanAttributes, EventAttributes

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


def test_milvus_single_vector_search(exporter, collection):
    insert_data(collection)

    query_vectors = [
        [random.uniform(-1, 1) for _ in range(5)],  # Random query vector for the search
    ]
    search_params = {"radius": 0.5, "metric_type": "COSINE", "index_type": "IVF_FLAT"}
    with pytest.raises(Exception) as exc_info:
        milvus.search(
            collection_name="random",
            data=query_vectors,
            anns_field="vector",
            search_params=search_params,
            output_fields=["color_tag"],
            limit="3", # wrong limit value
            timeout=10,
        )

    # Print the exception message if you want
    print(f"Caught expected exception: {exc_info.value}")

    # Get finished spans
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.search")

    # Print span name
    print(f"Span name: {span.name}")

    # Print span attributes
    print("Attributes:")
    for key, value in span.attributes.items():
        print(f"  {key}: {value}")

    # Print span status
    print("Status:")
    print(f"  Status code: {span.status.status_code}")
    print(f"  Description: {span.status.description}")

    # Print span events
    print("Events:")
    for event in span.events:
        print(f"  Event name: {event.name}")
        print(f"  Attributes: {event.attributes}")
    # Check the span attributes related to search
    assert span.attributes.get("error.type") == "ParamError"
