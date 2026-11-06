"""Closing a streamed response early must end the span and release the HTTP response.

These tests talk to an in-process ``httpx.MockTransport`` standing in for the
Ollama server, so they need no cassettes.
"""

import asyncio
import gc
import json
import logging

import httpx
import ollama
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import Meters, SpanAttributes
from opentelemetry.trace import StatusCode


class _Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    """NDJSON body that records whether the client closed it."""

    def __init__(self, lines, closed):
        self._lines = lines
        self._closed = closed

    def __iter__(self):
        yield from self._lines

    async def __aiter__(self):
        for line in self._lines:
            yield line

    def close(self):
        self._closed.append(True)

    async def aclose(self):
        self._closed.append(True)


@pytest.fixture
def server():
    """A stand-in Ollama server streaming a 3-part answer.

    ``server.requests`` lists the paths requested and ``server.closed`` is
    non-empty once the client has closed a response body.
    """

    class Server:
        requests = []
        closed = []

        def __call__(self, request):
            self.requests.append(request.url.path)
            chat = request.url.path.endswith("/api/chat")

            def part(text, done):
                payload = {
                    "model": "m",
                    "created_at": "2026-01-01T00:00:00Z",
                    "done": done,
                }
                if done:
                    payload["done_reason"] = "stop"
                    payload["prompt_eval_count"] = 3
                    payload["eval_count"] = 2
                if chat:
                    payload["message"] = {"role": "assistant", "content": text}
                else:
                    payload["response"] = text
                return (json.dumps(payload) + "\n").encode()

            lines = [part(text, False) for text in ("Hel", "lo", "!")]
            lines.append(part("", True))
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                stream=_Body(lines, self.closed),
            )

    server = Server()
    server.requests = []
    server.closed = []
    return server


@pytest.fixture
def client(server):
    return ollama.Client(host="http://mock", transport=httpx.MockTransport(server))


@pytest.fixture
def async_client(server):
    return ollama.AsyncClient(
        host="http://mock", transport=httpx.MockTransport(server)
    )


def _chat_kwargs():
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}


def test_close_before_reading_ends_span(instrument_legacy, client, span_exporter):
    stream = client.generate(model="m", prompt="hi", stream=True)

    stream.close()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["ollama.completion"]
    # a closed stream behaves like an exhausted one
    assert list(stream) == []


def test_close_is_idempotent(instrument_legacy, client, span_exporter, caplog):
    stream = client.generate(model="m", prompt="hi", stream=True)

    with caplog.at_level(logging.WARNING):
        stream.close()
        stream.close()
        del stream
        gc.collect()

    assert len(span_exporter.get_finished_spans()) == 1
    assert "ended span" not in caplog.text


def test_dropped_before_reading_ends_span(instrument_legacy, client, span_exporter):
    client.generate(model="m", prompt="hi", stream=True)
    gc.collect()

    assert len(span_exporter.get_finished_spans()) == 1


def test_close_after_partial_read_closes_response(
    instrument_legacy, client, server, span_exporter
):
    stream = client.chat(**_chat_kwargs())
    next(stream)
    assert not server.closed

    stream.close()

    assert server.closed
    assert [span.name for span in span_exporter.get_finished_spans()] == ["ollama.chat"]


def test_full_iteration_is_unchanged(
    instrument_legacy, client, server, span_exporter, caplog
):
    with caplog.at_level(logging.WARNING):
        stream = client.chat(**_chat_kwargs())
        content = "".join(part["message"]["content"] for part in stream)
        stream.close()

    assert content == "Hello!"
    assert server.closed
    assert len(span_exporter.get_finished_spans()) == 1
    assert "ended span" not in caplog.text


@pytest.mark.asyncio
async def test_async_aclose_before_reading_ends_span(
    instrument_legacy, async_client, span_exporter
):
    stream = await async_client.chat(**_chat_kwargs())

    await stream.aclose()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["ollama.chat"]
    assert [part async for part in stream] == []


@pytest.mark.asyncio
async def test_async_aclose_is_idempotent(
    instrument_legacy, async_client, span_exporter, caplog
):
    stream = await async_client.chat(**_chat_kwargs())

    with caplog.at_level(logging.WARNING):
        await stream.aclose()
        await stream.aclose()
        del stream
        gc.collect()

    assert len(span_exporter.get_finished_spans()) == 1
    assert "ended span" not in caplog.text


@pytest.mark.asyncio
async def test_async_dropped_before_reading_ends_span(
    instrument_legacy, async_client, span_exporter
):
    await async_client.chat(**_chat_kwargs())
    gc.collect()

    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_aclose_after_partial_read_closes_response(
    instrument_legacy, async_client, server, span_exporter
):
    stream = await async_client.chat(**_chat_kwargs())
    await stream.__anext__()
    assert not server.closed

    await stream.aclose()

    # closed right away, not when the stream is garbage-collected
    assert server.closed
    assert [span.name for span in span_exporter.get_finished_spans()] == ["ollama.chat"]


@pytest.mark.asyncio
async def test_async_full_iteration_is_unchanged(
    instrument_legacy, async_client, server, span_exporter, caplog
):
    with caplog.at_level(logging.WARNING):
        stream = await async_client.chat(**_chat_kwargs())
        content = "".join([part["message"]["content"] async for part in stream])
        await stream.aclose()

    assert content == "Hello!"
    assert server.closed
    assert len(span_exporter.get_finished_spans()) == 1
    assert "ended span" not in caplog.text


def _completion(span):
    return span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")


def test_break_at_done_records_the_response(instrument_legacy, client, span_exporter):
    """Stopping at the final ``done`` record is a common pattern, and the app
    already holds the whole answer there - the span must record it."""
    stream = client.chat(**_chat_kwargs())

    text = ""
    for chunk in stream:
        text += chunk.message.content
        if chunk.done:
            break
    stream.close()

    (span,) = span_exporter.get_finished_spans()
    assert text == "Hello!"
    assert _completion(span) == "Hello!"
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 5


def test_break_at_done_matches_full_iteration(instrument_legacy, client, span_exporter):
    for chunk in client.chat(**_chat_kwargs()):
        pass
    for chunk in client.chat(**_chat_kwargs()):
        if chunk.done:
            break
    gc.collect()

    full, broke_out = span_exporter.get_finished_spans()
    assert dict(broke_out.attributes) == dict(full.attributes)


def test_break_mid_stream_records_what_was_received(
    instrument_legacy, client, span_exporter
):
    stream = client.chat(**_chat_kwargs())

    text = ""
    for chunk in stream:
        text += chunk.message.content
        break
    stream.close()

    (span,) = span_exporter.get_finished_spans()
    assert text == "Hel"
    # what the app received, not an empty completion
    assert _completion(span) == "Hel"


@pytest.mark.asyncio
async def test_async_break_at_done_records_the_response(
    instrument_legacy, async_client, span_exporter
):
    stream = await async_client.chat(**_chat_kwargs())

    text = ""
    async for chunk in stream:
        text += chunk.message.content
        if chunk.done:
            break
    await stream.aclose()

    (span,) = span_exporter.get_finished_spans()
    assert text == "Hello!"
    assert _completion(span) == "Hello!"
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 5


@pytest.mark.asyncio
async def test_async_break_mid_stream_records_what_was_received(
    instrument_legacy, async_client, span_exporter
):
    stream = await async_client.chat(**_chat_kwargs())

    text = ""
    async for chunk in stream:
        text += chunk.message.content
        break
    await stream.aclose()

    (span,) = span_exporter.get_finished_spans()
    assert text == "Hel"
    assert _completion(span) == "Hel"


_USAGE_ATTRIBUTES = (
    SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
    GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
)


def _metric_points(reader, name):
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [
        point
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def _token_points(reader):
    return _metric_points(reader, Meters.LLM_TOKEN_USAGE)


def test_break_mid_stream_records_no_usage(
    instrument_legacy, client, span_exporter, reader
):
    """Token counts arrive only on the ``done`` record; a stream stopped
    before it must not report 0 tokens."""
    for chunk in client.chat(**_chat_kwargs()):
        break
    gc.collect()

    (span,) = span_exporter.get_finished_spans()
    assert _completion(span) == "Hel"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "m"
    for name in _USAGE_ATTRIBUTES:
        assert name not in span.attributes
    assert _token_points(reader) == []


def test_break_at_done_records_usage_metrics(instrument_legacy, client, reader):
    for chunk in client.chat(**_chat_kwargs()):
        if chunk.done:
            break
    gc.collect()

    assert sorted(point.sum for point in _token_points(reader)) == [2, 3]


@pytest.mark.asyncio
async def test_async_break_mid_stream_records_no_usage(
    instrument_legacy, async_client, span_exporter, reader
):
    stream = await async_client.chat(**_chat_kwargs())
    async for chunk in stream:
        break
    await stream.aclose()

    (span,) = span_exporter.get_finished_spans()
    assert _completion(span) == "Hel"
    for name in _USAGE_ATTRIBUTES:
        assert name not in span.attributes
    assert _token_points(reader) == []



def test_break_at_done_records_time_to_generate(instrument_legacy, client, reader):
    for chunk in client.chat(**_chat_kwargs()):
        if chunk.done:
            break
    gc.collect()

    assert _metric_points(reader, Meters.LLM_STREAMING_TIME_TO_GENERATE)


def test_break_mid_stream_records_no_time_to_generate(
    instrument_legacy, client, reader
):
    """A truncated generation would pull the histogram low."""
    for chunk in client.chat(**_chat_kwargs()):
        break
    gc.collect()

    assert _metric_points(reader, Meters.LLM_STREAMING_TIME_TO_GENERATE) == []


class _StallingBody(httpx.AsyncByteStream):
    """Sends two chat parts, then never sends the rest."""

    async def __aiter__(self):
        for text in ("Hel", "lo"):
            payload = {
                "model": "m",
                "created_at": "2026-01-01T00:00:00Z",
                "done": False,
                "message": {"role": "assistant", "content": text},
            }
            yield (json.dumps(payload) + "\n").encode()
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_async_cancelled_stream_records_what_was_received(
    instrument_legacy, span_exporter
):
    """A timeout or a client disconnecting from a streaming endpoint cancels
    the task reading the stream."""
    received = []
    client = ollama.AsyncClient(
        host="http://mock",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=_StallingBody())
        ),
    )

    async def consume():
        async for chunk in await client.chat(**_chat_kwargs()):
            received.append(chunk.message.content)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(consume(), timeout=0.2)

    (span,) = span_exporter.get_finished_spans()
    assert "".join(received) == "Hello"
    assert _completion(span) == "Hello"
    assert span.attributes[ERROR_TYPE] == "CancelledError"
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_cancellation_propagates_to_the_caller(
    instrument_legacy, span_exporter
):
    """Recording a cancelled stream must not swallow the cancellation: the
    awaiting caller has to see CancelledError, or cooperative cancellation
    breaks for everything upstream."""
    received = []
    two_parts_read = asyncio.Event()
    client = ollama.AsyncClient(
        host="http://mock",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=_StallingBody())
        ),
    )

    async def consume():
        async for chunk in await client.chat(**_chat_kwargs()):
            received.append(chunk.message.content)
            if len(received) == 2:
                two_parts_read.set()
        return "stream ended without cancellation"

    task = asyncio.create_task(consume())
    await two_parts_read.wait()
    # Let the consumer go back to awaiting the (stalled) next part.
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()

    (span,) = span_exporter.get_finished_spans()
    assert _completion(span) == "Hello"
    assert span.attributes[ERROR_TYPE] == "CancelledError"
    assert span.status.status_code == StatusCode.ERROR
