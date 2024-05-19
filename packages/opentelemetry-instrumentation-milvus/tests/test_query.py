import pytest
import pymilvus
import random

milvus = pymilvus.MilvusClient(uri="http://localhost:19530")


@pytest.fixture
def collection():
    collection_name = "Colors"
    milvus.create_collection(collection_name=collection_name, dimension=5)
    yield collection_name
    milvus.release_collection(collection_name=collection_name)
    milvus.drop_collection(collection_name=collection_name)


def insert_data(collection, create_partition=False):
    colors = ["green", "blue", "yellow", "red", "black", "white", "purple", "pink", "orange", "grey"]
    data = [{
        "id": i,
        "vector": [random.uniform(-1, 1) for _ in range(5)],
        "color": random.choice(colors),
        "tag": random.randint(1000, 9999)
    } for i in range(1000)]
    data += [
        {
            "id": 1000,
            "vector": [random.uniform(-1, 1) for _ in range(5)],
            "color": "brown",
            "tag": 1234
        },
        {
            "id": 1001,
            "vector": [random.uniform(-1, 1) for _ in range(5)],
            "color": "brown",
            "tag": 5678
        },
        {
            "id": 1002,
            "vector": [random.uniform(-1, 1) for _ in range(5)],
            "color": "brown",
            "tag": 9101
        }
    ]
    for i in data:
        i["color_tag"] = "{}_{}".format(i["color"], i["tag"])
    milvus.insert(collection_name=collection, data=data)

    if create_partition:
        milvus.create_partition(collection_name=collection, partition_name="partitionA")
        part_data = [
            {
                "id": 1003,
                "vector": [random.uniform(-1, 1) for _ in range(5)],
                "color": "crimson",
                "tag": 3489
            },
            {
                "id": 1004,
                "vector": [random.uniform(-1, 1) for _ in range(5)],
                "color": "crimson",
                "tag": 6453
            }
        ]
        milvus.insert(collection_name=collection,
                      data=part_data,
                      partition_name="partitionA"
                      )


def test_milvus_insert(exporter, collection):
    insert_data(collection)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.insert")

    assert span.attributes.get("db.system") == "milvus"
    assert span.attributes.get("db.operation") == "insert"
    assert span.attributes.get("db.milvus.insert.collection_name") == "Colors"
    assert span.attributes.get("db.milvus.insert.data_count") == 1003


def test_milvus_query(exporter, collection):
    insert_data(collection)
    milvus.query(
        collection_name=collection,
        filter='color == "brown"',
        output_fields=["color_tag"],
        limit=3
    )
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.query")

    assert span.attributes.get("db.system") == "milvus"
    assert span.attributes.get("db.operation") == "query"
    assert span.attributes.get("db.milvus.query.collection_name") == collection
    assert span.attributes.get("db.milvus.query.filter") == 'color == "brown"'
    assert span.attributes.get("db.milvus.query.output_fields_count") == 1
    assert span.attributes.get("db.milvus.query.limit") == 3

    events = span.events
    for event in events:
        assert event.name == "db.query.result"
        tag = event.attributes.get("color_tag")
        _id = event.attributes.get("id")
        assert isinstance(tag, str)
        assert isinstance(_id, int)


def test_milvus_query_partition(exporter, collection):
    insert_data(collection, create_partition=True)
    milvus.query(
        collection_name=collection,
        filter='color == "crimson"',
        output_fields=["color_tag"],
        partition_names=["partitionA"],
        limit=2
    )
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "milvus.query")

    assert span.attributes.get("db.system") == "milvus"
    assert span.attributes.get("db.operation") == "query"
    assert span.attributes.get("db.milvus.query.collection_name") == collection
    assert span.attributes.get("db.milvus.query.filter") == 'color == "crimson"'
    assert span.attributes.get("db.milvus.query.output_fields_count") == 1
    assert span.attributes.get("db.milvus.query.limit") == 2
    assert span.attributes.get("db.milvus.query.partition_names_count") == 1
    events = span.events
    for event in events:
        assert event.name == "db.query.result"
        tag = event.attributes.get("color_tag")
        _id = event.attributes.get("id")
        assert isinstance(tag, str)
        assert isinstance(_id, int)
