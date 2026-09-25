from unittest.mock import patch

import httpx
import pytest
from opentelemetry.trace.status import StatusCode


def test_mistralai_error_sets_span_status(instrument_legacy, mistralai_client, span_exporter):
    with patch("httpx.Client.send", side_effect=Exception("API connection error")):
        with pytest.raises(Exception, match="API connection error"):
            mistralai_client.chat.complete(
                model="mistral-tiny",
                messages=[{"role": "user", "content": "Tell me a joke"}],
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "API connection error" in span.status.description
    assert span.attributes.get("error.type") == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "API connection error" in events[0].attributes["exception.message"]


@pytest.mark.asyncio
async def test_async_mistralai_error_sets_span_status(instrument_legacy, mistralai_async_client, span_exporter):
    with patch("httpx.AsyncClient.send", side_effect=Exception("Async API error")):
        with pytest.raises(Exception, match="Async API error"):
            await mistralai_async_client.chat.complete_async(
                model="mistral-tiny",
                messages=[{"role": "user", "content": "Tell me a joke"}],
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "Async API error" in span.status.description
    assert span.attributes.get("error.type") == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1


_FIRST_EVENT = (
    b'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"mistral-tiny",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
)


class _DroppedSyncStream(httpx.SyncByteStream):
    def __iter__(self):
        yield _FIRST_EVENT
        raise httpx.ReadError("connection dropped mid-stream")


class _DroppedAsyncStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield _FIRST_EVENT
        raise httpx.ReadError("connection dropped mid-stream")


def _assert_stream_error_span(span_exporter):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "connection dropped mid-stream" in span.status.description
    assert span.attributes.get("error.type") == "ReadError"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1


def test_mistralai_stream_error_sets_span_status(instrument_legacy, mistralai_client, span_exporter):
    def send(self, request, **kwargs):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DroppedSyncStream(),
            request=request,
        )

    with patch("httpx.Client.send", send):
        stream = mistralai_client.chat.stream(
            model="mistral-tiny",
            messages=[{"role": "user", "content": "Tell me a joke"}],
        )
        received = 0
        with pytest.raises(httpx.ReadError):
            for _ in stream:
                received += 1
        assert received == 1

    _assert_stream_error_span(span_exporter)


@pytest.mark.asyncio
async def test_async_mistralai_stream_error_sets_span_status(
    instrument_legacy, mistralai_async_client, span_exporter
):
    async def send(self, request, **kwargs):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DroppedAsyncStream(),
            request=request,
        )

    with patch("httpx.AsyncClient.send", send):
        stream = await mistralai_async_client.chat.stream_async(
            model="mistral-tiny",
            messages=[{"role": "user", "content": "Tell me a joke"}],
        )
        received = 0
        with pytest.raises(httpx.ReadError):
            async for _ in stream:
                received += 1
        assert received == 1

    _assert_stream_error_span(span_exporter)
