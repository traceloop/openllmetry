import pytest
from qdrant_client import QdrantClient, models
from opentelemetry.semconv_ai import SpanAttributes


# This fixture returns an empty in-memory QdrantClient instance for each test
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

    assert span.attributes.get("qdrant.upsert.collection_name") == COLLECTION_NAME
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

    assert span.attributes.get("qdrant.upload_collection.collection_name") == COLLECTION_NAME
    assert span.attributes.get("qdrant.upload_collection.points_count") == 3


def query_points(qdrant: QdrantClient):
    qdrant.query_points(COLLECTION_NAME, query=[0.1] * EMBEDDING_DIM, limit=1)


def test_qdrant_query_points(exporter, qdrant):
    upload_collection(qdrant)
    query_points(qdrant)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "qdrant.query_points")

    assert span.attributes.get("qdrant.query_points.collection_name") == COLLECTION_NAME
    assert span.attributes.get(SpanAttributes.VECTOR_DB_QUERY_TOP_K) == 1


def query_batch_points(qdrant: QdrantClient):
    qdrant.query_batch_points(
        COLLECTION_NAME,
        requests=[
            models.QueryRequest(query=[0.1] * EMBEDDING_DIM, limit=10),
            models.QueryRequest(query=[0.2] * EMBEDDING_DIM, limit=5),
            models.QueryRequest(query=[0.42] * EMBEDDING_DIM, limit=2),
            models.QueryRequest(query=[0.21] * EMBEDDING_DIM, limit=1),
        ],
    )


def test_qdrant_query_batch_points(exporter, qdrant):
    upload_collection(qdrant)
    query_batch_points(qdrant)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "qdrant.query_batch_points")

    assert span.attributes.get("qdrant.query_batch_points.collection_name") == COLLECTION_NAME
    assert span.attributes.get("qdrant.query_batch_points.requests_count") == 4
