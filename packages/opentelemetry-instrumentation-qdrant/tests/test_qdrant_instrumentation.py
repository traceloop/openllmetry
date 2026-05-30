import asyncio

import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient, models
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


# --- AsyncQdrantClient coverage for query_points / query_batch_points ---
#
# The sync client's query_points / query_batch_points are already wrapped, but
# AsyncQdrantClient was missing the equivalents in async_qdrant_client_methods.json,
# so async users got no spans for those calls. These tests pin the parity.

async def _async_setup(client: AsyncQdrantClient) -> None:
    # `AsyncQdrantClient.upload_collection` runs through a ThreadPoolExecutor
    # internally and returns None instead of an awaitable, so set up the
    # fixture data with `upsert` which is genuinely async.
    await client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM, distance=models.Distance.COSINE
        ),
    )
    await client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(id=1, vector=[0.1] * EMBEDDING_DIM, payload={"name": "Paul"}),
            models.PointStruct(id=2, vector=[0.2] * EMBEDDING_DIM, payload={"name": "John"}),
            models.PointStruct(id=3, vector=[0.3] * EMBEDDING_DIM, payload={}),
        ],
    )


def test_qdrant_async_query_points(exporter):
    async def run() -> None:
        client = AsyncQdrantClient(location=":memory:")
        await _async_setup(client)
        await client.query_points(
            collection_name=COLLECTION_NAME,
            query=[0.1] * EMBEDDING_DIM,
            limit=1,
        )

    asyncio.run(run())

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "qdrant.query_points")

    assert (
        span.attributes.get(SpanAttributes.QDRANT_SEARCH_COLLECTION_NAME)
        == COLLECTION_NAME
    )
    assert span.attributes.get(SpanAttributes.VECTOR_DB_QUERY_TOP_K) == 1


def test_qdrant_async_query_batch_points(exporter):
    async def run() -> None:
        client = AsyncQdrantClient(location=":memory:")
        await _async_setup(client)
        await client.query_batch_points(
            collection_name=COLLECTION_NAME,
            requests=[
                models.QueryRequest(query=[0.1] * EMBEDDING_DIM, limit=10),
                models.QueryRequest(query=[0.2] * EMBEDDING_DIM, limit=5),
                models.QueryRequest(query=[0.42] * EMBEDDING_DIM, limit=2),
                models.QueryRequest(query=[0.21] * EMBEDDING_DIM, limit=1),
            ],
        )

    asyncio.run(run())

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "qdrant.query_batch_points")

    assert (
        span.attributes.get(SpanAttributes.QDRANT_SEARCH_BATCH_COLLECTION_NAME)
        == COLLECTION_NAME
    )
    assert span.attributes.get(SpanAttributes.QDRANT_SEARCH_BATCH_REQUESTS_COUNT) == 4
