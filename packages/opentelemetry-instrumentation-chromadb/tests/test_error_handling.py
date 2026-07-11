from unittest.mock import patch

import chromadb
import pytest
from opentelemetry.trace.status import StatusCode


chroma = chromadb.EphemeralClient()


@pytest.fixture
def collection(exporter):
    col = chroma.create_collection(name="test_errors")
    yield col
    chroma.delete_collection(name="test_errors")


def test_chromadb_add_error_sets_span_status_and_error_type(exporter, collection):
    # Patch the internal client method so the wrapt wrapper (and thus the span) still fires
    with patch.object(
        type(collection._client), "_add", side_effect=Exception("DB write error")
    ):
        with pytest.raises(Exception, match="DB write error"):
            collection.add(ids=["id1"], documents=["some document"])

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert "DB write error" in span.status.description
    assert span.attributes.get("error.type") == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "DB write error" in events[0].attributes["exception.message"]


def test_chromadb_query_error_sets_span_status_and_error_type(exporter, collection):
    collection.add(ids=["id1"], documents=["some document"])

    with patch.object(
        type(collection._client), "_query", side_effect=Exception("Query failed")
    ):
        with pytest.raises(Exception, match="Query failed"):
            collection.query(query_texts=["test query"], n_results=1)

    spans = exporter.get_finished_spans()
    error_spans = [s for s in spans if s.status.status_code == StatusCode.ERROR]
    assert len(error_spans) == 1
    span = error_spans[0]

    assert "Query failed" in span.status.description
    assert span.attributes.get("error.type") == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
