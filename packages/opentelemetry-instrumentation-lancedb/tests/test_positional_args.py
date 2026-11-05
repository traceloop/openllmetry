"""Regression tests: span attributes must be captured for positional arguments.

LanceDB documents ``table.add(data)``, ``table.search(query)`` and
``table.delete(where)`` as positional calls, so the instrumentation must not
rely on ``kwargs`` alone. These tests use a local, temporary LanceDB.
"""

import lancedb
import pytest
from opentelemetry.semconv_ai import SpanAttributes

ROWS = [
    {"vector": [1.3, 1.4], "item": "fizz", "price": 100.0},
    {"vector": [9.5, 56.2], "item": "buzz", "price": 200.0},
]


@pytest.fixture
def table(tmp_path):
    db = lancedb.connect(str(tmp_path))
    return db.create_table("positional_table", data=ROWS)


def _span(exporter, name):
    return next(s for s in exporter.get_finished_spans() if s.name == name)


def test_add_positional_args(exporter, table):
    exporter.clear()
    table.add(ROWS)

    span = _span(exporter, "lancedb.add")
    assert span.attributes.get(SpanAttributes.MILVUS_INSERT_DATA_COUNT) == 2


def test_search_positional_args(exporter, table):
    exporter.clear()
    table.search([100, 100]).limit(2).to_pandas()

    span = _span(exporter, "lancedb.search")
    assert span.attributes.get(SpanAttributes.MILVUS_SEARCH_FILTER) == "[100, 100]"


def test_delete_positional_args(exporter, table):
    exporter.clear()
    table.delete("item = 'fizz'")

    span = _span(exporter, "lancedb.delete")
    assert span.attributes.get(SpanAttributes.CHROMADB_DELETE_WHERE) == "item = 'fizz'"
