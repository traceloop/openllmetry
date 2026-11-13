"""Regression tests: span attributes must be captured for positional arguments.

``MilvusClient.insert(collection_name, data)``, ``search(collection_name, data)``
and ``delete(collection_name, ids)`` are valid positional calls, so the
instrumentation must not rely on ``kwargs`` alone. These tests use Milvus Lite
with a temporary local database, like the other tests in this package.
"""

import pymilvus
import pytest
from opentelemetry.semconv_ai import SpanAttributes

COLLECTION = "PositionalColors"
ROWS = [
    {"id": 0, "vector": [0.1, 0.2, 0.3, 0.4, 0.5]},
    {"id": 1, "vector": [0.5, 0.4, 0.3, 0.2, 0.1]},
]


@pytest.fixture
def client(tmp_path):
    milvus = pymilvus.MilvusClient(uri=str(tmp_path / "positional.db"))
    milvus.create_collection(collection_name=COLLECTION, dimension=5)
    yield milvus
    milvus.drop_collection(collection_name=COLLECTION)


def _span(exporter, name):
    return next(s for s in exporter.get_finished_spans() if s.name == name)


def test_insert_positional_args(exporter, client):
    exporter.clear()
    client.insert(COLLECTION, ROWS)

    span = _span(exporter, "milvus.insert")
    assert span.attributes.get(SpanAttributes.MILVUS_INSERT_COLLECTION_NAME) == COLLECTION
    assert span.attributes.get(SpanAttributes.MILVUS_INSERT_DATA_COUNT) == 2


def test_search_positional_args(exporter, client):
    client.insert(COLLECTION, ROWS)
    exporter.clear()
    client.search(COLLECTION, [[0.1, 0.2, 0.3, 0.4, 0.5]], limit=2)

    span = _span(exporter, "milvus.search")
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_COLLECTION_NAME) == COLLECTION
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_DATA_COUNT) == 1


def test_delete_positional_args(exporter, client):
    client.insert(COLLECTION, ROWS)
    exporter.clear()
    client.delete(COLLECTION, [0, 1])

    span = _span(exporter, "milvus.delete")
    assert span.attributes.get(SpanAttributes.MILVUS_DELETE_COLLECTION_NAME) == COLLECTION
    assert span.attributes.get(SpanAttributes.MILVUS_DELETE_IDS_COUNT) == 2
