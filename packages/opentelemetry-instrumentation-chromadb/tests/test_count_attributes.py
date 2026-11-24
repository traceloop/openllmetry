"""Regression tests: chromadb count attributes must survive sized-but-truthy-ambiguous inputs.

numpy arrays are a supported chromadb input type (``add``/``query`` accept them),
but ``count_or_none`` used a bare truthiness check, which raises on arrays with
more than one element. The exception was swallowed by ``dont_throw``, so the
count attribute and every attribute set after it in the same helper disappeared.
"""

import chromadb
import numpy as np
import pytest

from opentelemetry.instrumentation.chromadb.wrapper import count_or_none
from opentelemetry.semconv_ai import SpanAttributes

chroma = chromadb.EphemeralClient()


@pytest.fixture
def collection():
    col = chroma.create_collection(name="test_count_attributes")
    yield col
    chroma.delete_collection(name="test_count_attributes")


def test_count_or_none_handles_sized_containers():
    assert count_or_none(None) is None
    assert count_or_none([]) is None
    assert count_or_none("abc") == 3
    assert count_or_none([1, 2, 3]) == 3
    # An array with more than one element raises on `if obj:`.
    assert count_or_none(np.array([[1.0, 0.0], [0.0, 1.0]])) == 2
    # Objects without a length are not countable, but must not raise either.
    assert count_or_none(object()) is None


def test_query_records_counts_for_numpy_embeddings(exporter, collection):
    try:
        collection.add(
            ids=["1", "2"],
            documents=["doc one", "doc two"],
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - old chromadb
        pytest.skip(f"installed chromadb rejects numpy embeddings: {exc}")

    collection.query(query_embeddings=np.array([[1.0, 0.0]]), n_results=1)

    spans = exporter.get_finished_spans()
    add_span = next(span for span in spans if span.name == "chroma.add")
    query_span = next(span for span in spans if span.name == "chroma.query")

    assert add_span.attributes.get(SpanAttributes.CHROMADB_ADD_IDS_COUNT) == 2
    assert add_span.attributes.get(SpanAttributes.CHROMADB_ADD_EMBEDDINGS_COUNT) == 2
    assert add_span.attributes.get(SpanAttributes.CHROMADB_ADD_DOCUMENTS_COUNT) == 2
    assert query_span.attributes.get(SpanAttributes.CHROMADB_QUERY_EMBEDDINGS_COUNT) == 1
    # Recorded after embeddings_count in the same helper: also lost before the fix.
    assert query_span.attributes.get(SpanAttributes.CHROMADB_QUERY_N_RESULTS) == 1
