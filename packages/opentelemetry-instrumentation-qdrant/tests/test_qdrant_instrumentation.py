import pytest
from qdrant_client import QdrantClient, models
from opentelemetry.semconv.ai import SpanAttributes


# This fixture returns an empty in-memroy QdrantClient instance for each test
@pytest.fixture
def qdrant():
    yield QdrantClient(location=":memory:")


COLLECTION_NAME = "test_collection"
EMBEDDING_DIM = 384


def upsert(qdrant: QdrantClient):
    qdrant.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )

    qdrant.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(id=1, vector=[0.1] * EMBEDDING_DIM, payload={"name": "Paul"}),
            models.PointStruct(id=2, vector=[0.2] * EMBEDDING_DIM, payload={"name": "John"}),
        ],
    )


def test_qdrant_upsert(exporter, qdrant):
    upsert(qdrant)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == f"{SpanAttributes.QDRANT_UPSERT}")

    assert span.attributes.get(f"{SpanAttributes.QDRANT_UPSERT}.collection_name") == COLLECTION_NAME
    assert span.attributes.get(f"{SpanAttributes.QDRANT_UPSERT}.points_count") == 2


def upload_collection(qdrant: QdrantClient):
    qdrant.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
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
    span = next(span for span in spans if span.name == f"{SpanAttributes.QDRANT_UPLOAD_COLLECTION}")

    assert (
        span.attributes.get(f"{SpanAttributes.QDRANT_UPLOAD_COLLECTION}.collection_name")
        == COLLECTION_NAME
    )
    assert span.attributes.get(f"{SpanAttributes.QDRANT_UPLOAD_COLLECTION}.points_count") == 3


def search(qdrant: QdrantClient):
    qdrant.search(COLLECTION_NAME, query_vector=[0.1] * EMBEDDING_DIM, limit=1)


def test_qdrant_searchs(exporter, qdrant):
    upload_collection(qdrant)
    search(qdrant)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == f"{SpanAttributes.QDRANT_SEARCH}")

    assert span.attributes.get(f"{SpanAttributes.QDRANT_SEARCH}.collection_name") == COLLECTION_NAME
    assert span.attributes.get(SpanAttributes.VECTOR_DB_QUERY_TOP_K) == 1


def search_batch(qdrant: QdrantClient):
    qdrant.search_batch(
        COLLECTION_NAME, requests=[models.SearchRequest(vector=[0.1] * EMBEDDING_DIM, limit=10),
                                   models.SearchRequest(vector=[0.2] * EMBEDDING_DIM, limit=5),
                                   models.SearchRequest(vector=[0.42] * EMBEDDING_DIM, limit=2),
                                   models.SearchRequest(vector=[0.21] * EMBEDDING_DIM, limit=1)]
    )


def test_qdrant_search(exporter, qdrant):
    upload_collection(qdrant)
    search_batch(qdrant)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == f"{SpanAttributes.QDRANT_SEARCH_BATCH}")

    assert span.attributes.get(f"{SpanAttributes.QDRANT_SEARCH_BATCH}.collection_name") == COLLECTION_NAME
    assert span.attributes.get(f"{SpanAttributes.QDRANT_SEARCH_BATCH}.requests_count") == 4
