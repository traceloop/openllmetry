"""Tests for ChromaDB query result events — issue #1870."""
import chromadb
import pytest

chroma = chromadb.EphemeralClient()


@pytest.fixture
def collection(exporter):
    col = chroma.create_collection(name="test_results")
    yield col
    chroma.delete_collection(name="test_results")


def test_chromadb_query_returns_all_chunks(exporter, collection):
    """query() with n_results=2 should produce 2 result events, not 1."""
    collection.add(
        ids=["1", "2", "3"],
        documents=["doc one", "doc two", "doc three"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
    )
    collection.query(query_embeddings=[[1.0, 0.0]], n_results=2)

    spans = exporter.get_finished_spans()
    query_span = next(s for s in spans if s.name == "chroma.query")
    result_events = [e for e in query_span.events if e.name == "db.query.result"]

    assert len(result_events) == 2, (
        f"Expected 2 result events, got {len(result_events)}"
    )


def test_chromadb_multi_query_returns_all_chunks(exporter, collection):
    """2 query embeddings × n_results=2 should produce 4 events total."""
    collection.add(
        ids=["1", "2", "3", "4"],
        documents=["doc one", "doc two", "doc three", "doc four"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.3, 0.7]],
    )
    collection.query(
        query_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        n_results=2,
    )

    spans = exporter.get_finished_spans()
    query_span = next(s for s in spans if s.name == "chroma.query")
    result_events = [e for e in query_span.events if e.name == "db.query.result"]

    assert len(result_events) == 4, (
        f"Expected 4 result events (2 queries × 2 results), got {len(result_events)}"
    )


def test_chromadb_query_result_events_contain_correct_data(exporter, collection):
    """Each result event should contain id, distance, document and metadata."""
    collection.add(
        ids=["doc-id-aaa", "doc-id-bbb"],
        documents=["doc one", "doc two"],
        metadatas=[{"source": "fileA"}, {"source": "fileB"}],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    collection.query(query_embeddings=[[1.0, 0.0]], n_results=2)

    spans = exporter.get_finished_spans()
    query_span = next(s for s in spans if s.name == "chroma.query")
    result_events = [e for e in query_span.events if e.name == "db.query.result"]

    assert len(result_events) == 2
    for event in result_events:
        assert "db.query.result.id" in event.attributes
        assert "db.query.result.distance" in event.attributes
        assert "db.query.result.document" in event.attributes
        assert "db.query.result.metadata" in event.attributes

    ids_recorded = {e.attributes["db.query.result.id"] for e in result_events}
    assert ids_recorded == {"doc-id-aaa", "doc-id-bbb"}
