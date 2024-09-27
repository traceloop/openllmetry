import pytest
import numpy as np
from redis import Redis
from redis.exceptions import ResponseError
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.aggregation import AggregateRequest
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField,
)


@pytest.fixture
def redis_client():
    yield Redis(host='localhost', port=6379, decode_responses=True)


EMBEDDING_DIM = 256


def prepare_data(redis_client):
    try:
        redis_client.ft("idx:test_vss").dropindex(True)
    except ResponseError:
        print("No such index")
    item = {"name": "test",
            "embeddings": [0.1] * 256}
    pipeline = redis_client.pipeline()
    pipeline.json().set(f"test:001", "$", item)
    res = pipeline.execute()
    assert False not in res


def create_index(redis_client):
    schema =  ( 
                TextField("$.name", no_stem=True, as_name="name"), 
                VectorField("$.embeddings",
                            "FLAT",
                            {
                                "TYPE": "FLOAT32",
                                "DIM": EMBEDDING_DIM,
                                "DISTANCE_METRIC": "COSINE",
                            },
                            as_name="vector",),
                )
    definition = IndexDefinition(prefix=["test:"], index_type=IndexType.JSON)
    res = redis_client.ft("idx:test_vss").create_index(fields=schema, definition=definition)
    assert "OK" in res


def test_redis_aggregate(redis_client, exporter):
    name = "test:001"
    path = "$"
    item = {"name": "test_name",
            "value": "test_value"}
    redis_client.json().set(name, path, item)
    schema = (
        TextField("$.name", no_stem=True, as_name="name"),
        TextField("$.value", no_stem=True, as_name="value"),
    )
    try:
        redis_client.ft("idx:test").dropindex(True)
    except ResponseError:
        print("No such index")
    definition = IndexDefinition(prefix=["test:"], index_type=IndexType.JSON)
    redis_client.ft("idx:test").create_index(fields=schema, definition=definition)
    query = "*"
    redis_client.ft("idx:test").aggregate(AggregateRequest(query).load())
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "redis.aggregate")
    assert span.attributes.get("redis.commands.aggregate.query") == query
    assert "redis.commands.aggregate.results" in span.attributes


def test_redis_create_index(redis_client, exporter):
    prepare_data(redis_client)
    create_index(redis_client)
    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "redis.create_index")
    assert "redis.create_index.definition" in span.attributes
    assert "redis.create_index.fields" in span.attributes


def test_redis_query(redis_client, exporter):
    prepare_data(redis_client)
    create_index(redis_client)
    query = "@name:test"
    res = redis_client.ft("idx:test_vss").search(Query(query))

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "redis.search")

    assert span.attributes.get("redis.commands.search.query") == query
    assert span.attributes.get("redis.commands.search.total") == 1
