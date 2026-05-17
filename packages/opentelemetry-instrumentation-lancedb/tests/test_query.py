import lancedb
import pytest
from opentelemetry.semconv_ai import SpanAttributes

db = lancedb.connect("data/sample-lancedb")


@pytest.fixture
def collection():
    data = [
        {"vector": [1.3, 1.4], "item": "fizz", "price": 100.0},
        {"vector": [9.5, 56.2], "item": "buzz", "price": 200.0},
    ]
    yield db.create_table("my_table", data=data)
    db.drop_table("my_table")


def add_data(collection):
    dataToAdd = [
        {"vector": [1.3, 1.4], "item": "fizz", "price": 100.0},
        {"vector": [9.5, 56.2], "item": "buzz", "price": 200.0},
    ]
    tbl = db.open_table("my_table")
    tbl.add(data=dataToAdd)


def test_lancedb_add(exporter, collection):
    exporter.clear()
    add_data(collection)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "lancedb.add")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "lancedb"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "add"
    assert span.attributes.get(SpanAttributes.MILVUS_INSERT_DATA_COUNT) == 2


def test_lancedb_search(exporter, collection):
    add_data(collection)
    tbl = db.open_table("my_table")
    tbl.search(query=[100, 100]).limit(2).to_pandas()

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "lancedb.search")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "lancedb"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "search"
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_FILTER) == "[100, 100]"


def test_lancedb_delete(exporter, collection):
    add_data(collection)
    tbl = db.open_table("my_table")
    tbl.delete(where='item = "fizz"')

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "lancedb.delete")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "lancedb"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "delete"
    assert span.attributes.get(SpanAttributes.CHROMADB_DELETE_WHERE) == 'item = "fizz"'
