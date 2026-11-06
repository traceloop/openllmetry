import pytest
from qdrant_client import QdrantClient, models
from opentelemetry.semconv_ai import SpanAttributes


# This fixture returns an empty in-memroy QdrantClient instance for each test
@pytest.fixture
def qdrant():
    yield QdrantClient(location=":memory:")


COLLECTION_NAME = "test_collection"
EMBEDDING_DIM = 384


def upsert(qdrant: QdrantClient):
    qdrant.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM, distance=models.Distance.COSINE
        ),
    )

    qdrant.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=1, vector=[0.1] * EMBEDDING_DIM, payload={"name": "Paul"}
            ),
            models.PointStruct(
                id=2, vector=[0.2] * EMBEDDING_DIM, payload={"name": "John"}
            ),
        ],
    )


def test_qdrant_upsert(exporter, qdrant):
    upsert(qdrant)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "qdrant.upsert")

    assert (
        span.attributes.get(SpanAttributes.QDRANT_UPSERT_COLLECTION_NAME)
        == COLLECTION_NAME
    )
    assert span.attributes.get(SpanAttributes.QDRANT_UPSERT_POINTS_COUNT) == 2


def upload_collection(qdrant: QdrantClient):
    qdrant.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM, distance=models.Distance.COSINE
        ),
    )

    qdrant.upload_collection(
        COLLECTION_NAME,
        vectors=[[0.1] * EMBEDDING_DIM, [0.2] * EMBEDDING_DIM, [0.3] * EMBEDDING_DIM],
        ids=[3, 21, 1],
        payload=[{"name": "Paul"}, {"name": "John"}, {}],
    )


def test_qdrant_upload_collection(exporter, qdrant):
    upload_collection(qdrant)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "qdrant.upload_collection")

    assert (
        span.attributes.get(SpanAttributes.QDRANT_UPLOAD_COLLECTION_NAME)
        == COLLECTION_NAME
    )
    assert span.attributes.get(SpanAttributes.QDRANT_UPLOAD_POINTS_COUNT) == 3


def search(qdrant: QdrantClient):
    qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=[0.1] * EMBEDDING_DIM,
        limit=1
    )


def test_qdrant_searchs(exporter, qdrant):
    upload_collection(qdrant)
    search(qdrant)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "qdrant.query_points")

    assert (
        span.attributes.get(SpanAttributes.QDRANT_SEARCH_COLLECTION_NAME)
        == COLLECTION_NAME
    )
    assert span.attributes.get(SpanAttributes.VECTOR_DB_QUERY_TOP_K) == 1


def search_batch(qdrant: QdrantClient):
    qdrant.query_batch_points(
        collection_name=COLLECTION_NAME,
        requests=[
            models.QueryRequest(query=[0.1] * EMBEDDING_DIM, limit=10),
            models.QueryRequest(query=[0.2] * EMBEDDING_DIM, limit=5),
            models.QueryRequest(query=[0.42] * EMBEDDING_DIM, limit=2),
            models.QueryRequest(query=[0.21] * EMBEDDING_DIM, limit=1),
        ],
    )


def test_qdrant_search(exporter, qdrant):
    upload_collection(qdrant)
    search_batch(qdrant)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "qdrant.query_batch_points")

    assert (
        span.attributes.get(SpanAttributes.QDRANT_SEARCH_BATCH_COLLECTION_NAME)
        == COLLECTION_NAME
    )
    assert span.attributes.get(SpanAttributes.QDRANT_SEARCH_BATCH_REQUESTS_COUNT) == 4


def test_wrapped_methods_exist_on_client_surface():
    """
    Regression test for https://github.com/traceloop/openllmetry/issues/3492.

    On qdrant-client 1.16 a handful of legacy methods (search, recommend,
    discover, upload_records and their batch variants) were removed. The
    instrumentor used to look them up unconditionally and blew up with an
    AttributeError before any user code had a chance to run.

    The runtime guards in _instrument / _uninstrument now skip missing
    methods silently, which keeps users unblocked but hides the drift. Walk
    the wrapped method tables directly against the installed client so
    CI fails loudly if the JSON drifts out of sync with qdrant-client
    again, and keep a full instrument/uninstrument cycle to make sure the
    wrapping itself still works end to end.
    """
    import qdrant_client
    from opentelemetry.instrumentation.qdrant import (
        QdrantInstrumentor,
        WRAPPED_METHODS,
    )

    missing = []
    for entry in WRAPPED_METHODS:
        obj_name = entry["object"]
        method_name = entry["method"]
        client_cls = getattr(qdrant_client, obj_name, None)
        if client_cls is None or not hasattr(client_cls, method_name):
            missing.append(f"{obj_name}.{method_name}")

    assert not missing, (
        "Wrapped methods no longer present on qdrant-client: "
        f"{missing}. Update the method JSON files under "
        "opentelemetry/instrumentation/qdrant/ to match."
    )

    instrumentor = QdrantInstrumentor()
    instrumentor.instrument()
    instrumentor.uninstrument()
