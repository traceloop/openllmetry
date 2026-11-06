import asyncio
import inspect
from contextlib import asynccontextmanager, suppress

import httpx
import pytest
from opentelemetry import context, trace
from opentelemetry.instrumentation.qdrant import QdrantInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse


COLLECTION_NAME = "async_test_collection"
POINTS = [models.PointStruct(id=1, vector=[0.1, 0.2])]
METHODS = ["upsert", "search", "search_batch"]


@asynccontextmanager
async def _client(**kwargs):
    """Yield an async client and close it even when the test raises or cancels."""
    client = AsyncQdrantClient(**kwargs)
    try:
        yield client
    finally:
        await client.close()


def _call(client, method):
    """Create a representative Qdrant operation without awaiting it."""
    if method == "upsert":
        return client.upsert(collection_name=COLLECTION_NAME, points=POINTS)
    if method == "search":
        return client.search(collection_name=COLLECTION_NAME, query_vector=[0.1, 0.2], limit=1)
    return client.search_batch(
        collection_name=COLLECTION_NAME,
        requests=[models.SearchRequest(vector=[0.1, 0.2], limit=1)],
    )


def _response(method):
    """Build a successful HTTP response with the result shape for the method."""
    if method == "upsert":
        result = {"operation_id": 1, "status": "completed"}
    else:
        result = [{"id": 1, "version": 0, "score": 0.9}]
        if method == "search_batch":
            result = [result]
    return httpx.Response(200, json={"result": result, "status": "ok", "time": 0.0})


def _spans(exporter, method):
    """Return finished Qdrant spans for the requested client method."""
    return [span for span in exporter.get_finished_spans() if span.name == f"qdrant.{method}"]


@pytest.mark.parametrize("method", METHODS)
def test_async_span_covers_await_and_preserves_parent(exporter, method):
    """Keep the Qdrant span current and open throughout the awaited request."""
    async def run():
        """Coordinate a pending request and verify its span hierarchy and result."""
        entered, release = asyncio.Event(), asyncio.Event()
        tracer = trace.get_tracer(__name__)
        active_spans = []

        async def handler(request):
            """Hold a child transport span open until the test releases the request."""
            active_spans.append(trace.get_current_span())
            with tracer.start_as_current_span("transport"):
                entered.set()
                await release.wait()
            return _response(method)

        async with _client(
            url="http://localhost:6333", transport=httpx.MockTransport(handler)
        ) as client:
            with tracer.start_as_current_span("caller") as caller:
                pending = _call(client, method)
                task = None
                try:
                    assert _spans(exporter, method) == []
                    assert not entered.is_set()
                    task = asyncio.create_task(pending)
                    await asyncio.wait_for(entered.wait(), timeout=2)
                    assert not task.done()
                    assert _spans(exporter, method) == []
                    assert active_spans[0].is_recording()
                    release.set()
                    result = await task
                finally:
                    release.set()
                    if task is None:
                        pending.close()
                    elif not task.done():
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task

                spans = _spans(exporter, method)
                assert len(spans) == 1
                span = spans[0]
                child = next(s for s in exporter.get_finished_spans() if s.name == "transport")
                assert span.parent.span_id == caller.get_span_context().span_id
                assert child.parent.span_id == span.context.span_id
                assert span.start_time <= child.start_time <= child.end_time <= span.end_time
                assert trace.get_current_span() is caller
                assert span.status.status_code == StatusCode.OK
                assert span.attributes[SpanAttributes.VECTOR_DB_VENDOR] == "Qdrant"
                assert span.attributes[f"qdrant.{method}.collection_name"] == COLLECTION_NAME
                if method == "upsert":
                    assert isinstance(result, models.UpdateResult)
                    assert result.status == models.UpdateStatus.COMPLETED
                    assert span.attributes[SpanAttributes.QDRANT_UPSERT_POINTS_COUNT] == 1
                elif method == "search":
                    assert result[0].id == 1
                    assert span.attributes[SpanAttributes.VECTOR_DB_QUERY_TOP_K] == 1
                else:
                    assert result[0][0].id == 1
                    assert span.attributes[SpanAttributes.QDRANT_SEARCH_BATCH_REQUESTS_COUNT] == 1

    asyncio.run(run())


@pytest.mark.parametrize("method", METHODS)
def test_async_exception_is_recorded_and_propagated(exporter, method):
    """Propagate awaited request errors and record exactly one exception event."""
    async def run():
        """Exercise a failed request and inspect its completed Qdrant span."""
        async def handler(request):
            """Return an HTTP error after yielding control to the event loop."""
            await asyncio.sleep(0)
            return httpx.Response(500, json={"status": {"error": "test failure"}})

        async with _client(
            url="http://localhost:6333", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(UnexpectedResponse):
                await _call(client, method)
        spans = _spans(exporter, method)
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        exceptions = [event for event in spans[0].events if event.name == "exception"]
        assert len(exceptions) == 1
        assert exceptions[0].attributes["exception.type"].endswith("UnexpectedResponse")

    asyncio.run(run())


def test_async_cancellation_closes_span_and_restores_context(exporter):
    """End cancelled operation spans and restore the caller's tracing context."""
    async def run():
        """Cancel a pending operation and verify span and context cleanup."""
        entered = asyncio.Event()
        operation_spans = []
        restored_contexts = []
        tracer = trace.get_tracer(__name__)

        async def handler(request):
            """Capture the active operation span and block until cancellation."""
            operation_spans.append(trace.get_current_span())
            entered.set()
            await asyncio.Event().wait()

        async with _client(
            url="http://localhost:6333", transport=httpx.MockTransport(handler)
        ) as client:
            with tracer.start_as_current_span("caller") as caller:
                async def operation():
                    """Capture the restored context while cancellation unwinds the call."""
                    try:
                        await _call(client, "upsert")
                    finally:
                        restored_contexts.append(trace.get_current_span())

                task = asyncio.create_task(operation())
                try:
                    await asyncio.wait_for(entered.wait(), timeout=2)
                finally:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task

                assert not operation_spans[0].is_recording()
                assert restored_contexts == [caller]
                assert trace.get_current_span() is caller
        spans = _spans(exporter, "upsert")
        assert len(spans) == 1
        assert spans[0].status.status_code != StatusCode.OK

    asyncio.run(run())


@pytest.mark.parametrize("method", METHODS)
def test_suppressed_async_call_is_awaited_without_a_span(exporter, method):
    """Await suppressed client calls normally without exporting Qdrant spans."""
    async def run():
        """Run a suppressed request and verify its execution and resolved result."""
        requests = []

        async def handler(request):
            """Track request execution independently of instrumentation suppression."""
            requests.append(request)
            return _response(method)

        async with _client(
            url="http://localhost:6333", transport=httpx.MockTransport(handler)
        ) as client:
            token = context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
            try:
                result = await _call(client, method)
            finally:
                context.detach(token)
        assert len(requests) == 1
        assert not inspect.isawaitable(result)
        assert _spans(exporter, method) == []

    asyncio.run(run())


def test_unawaited_async_call_does_not_create_a_span(exporter):
    """Avoid exporting spans for coroutine objects that never execute."""
    async def run():
        """Close an unawaited operation and verify that no span was created."""
        async with _client(location=":memory:") as client:
            pending = _call(client, "upsert")
            pending.close()
        assert _spans(exporter, "upsert") == []

    asyncio.run(run())


def test_async_empty_result_is_preserved(exporter):
    """Preserve empty results and the existing UNSET status for falsey results."""
    async def run():
        """Await an empty search response and inspect its span status."""
        async def handler(request):
            """Return a valid search response containing no matches."""
            return httpx.Response(200, json={"result": [], "status": "ok", "time": 0.0})

        async with _client(
            url="http://localhost:6333", transport=httpx.MockTransport(handler)
        ) as client:
            assert await _call(client, "search") == []
        spans = _spans(exporter, "search")
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.UNSET

    asyncio.run(run())


@pytest.mark.parametrize("method", ["upload_points", "upload_records", "upload_collection"])
def test_async_client_sync_upload_methods_remain_sync(exporter, method):
    """Keep synchronous upload methods synchronous on the async client."""
    async def run():
        """Upload to an in-memory collection and verify immediate completion."""
        async with _client(location=":memory:") as client:
            await client.create_collection(
                COLLECTION_NAME, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
            )
            if method == "upload_collection":
                result = client.upload_collection(COLLECTION_NAME, vectors=[[0.1, 0.2]], ids=[1])
            elif method == "upload_points":
                result = client.upload_points(COLLECTION_NAME, points=POINTS)
            else:
                result = client.upload_records(
                    COLLECTION_NAME, records=[models.Record(id=1, vector=[0.1, 0.2])]
                )
            assert result is None
            assert len(_spans(exporter, method)) == 1
            assert (await client.count(COLLECTION_NAME)).count == 1

    asyncio.run(run())


def test_uninstrument_restores_sync_and_async_methods(exporter):
    """Restore original methods for both coroutine and synchronous wrappers."""
    originals = {
        (client, method): getattr(client, method).__wrapped__
        for client, method in [
            (QdrantClient, "upsert"),
            (AsyncQdrantClient, "upsert"),
            (AsyncQdrantClient, "upload_collection"),
        ]
    }
    instrumentor = QdrantInstrumentor()
    instrumentor.uninstrument()
    try:
        for (client, method), original in originals.items():
            assert getattr(client, method) is original
    finally:
        instrumentor.instrument()
