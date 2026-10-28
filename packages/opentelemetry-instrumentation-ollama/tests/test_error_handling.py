from unittest.mock import patch

import pytest
from opentelemetry.trace.status import StatusCode


def test_ollama_error_sets_span_status(instrument_legacy, ollama_client, span_exporter):
    with patch("httpx.Client.send", side_effect=Exception("Connection refused")):
        with pytest.raises(Exception, match="Connection refused"):
            ollama_client.chat(
                model="llama3",
                messages=[{"role": "user", "content": "Tell me a joke"}],
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "Connection refused" in span.status.description
    assert span.attributes.get("error.type") == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "Connection refused" in events[0].attributes["exception.message"]


@pytest.mark.asyncio
async def test_async_ollama_error_sets_span_status(instrument_legacy, ollama_client_async, span_exporter):
    with patch("httpx.AsyncClient.send", side_effect=Exception("Async connection error")):
        with pytest.raises(Exception, match="Async connection error"):
            await ollama_client_async.chat(
                model="llama3",
                messages=[{"role": "user", "content": "Tell me a joke"}],
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "Async connection error" in span.status.description
    assert span.attributes.get("error.type") == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "Async connection error" in events[0].attributes["exception.message"]


def test_ollama_streaming_iteration_error_sets_span_status(
    instrument_legacy, tracer_provider, span_exporter
):
    """Errors raised during stream iteration must still finalize the span
    with ERROR status, error.type, and an exception event."""
    from opentelemetry.instrumentation.ollama import _accumulate_streaming_response
    from opentelemetry.semconv_ai import LLMRequestTypeValues

    span = tracer_provider.get_tracer(__name__).start_span("ollama.chat")

    def failing_stream():
        raise Exception("Mid-stream failure")
        yield  # unreachable, makes this a generator

    gen = _accumulate_streaming_response(
        span, None, None, LLMRequestTypeValues.CHAT, failing_stream()
    )
    with pytest.raises(Exception, match="Mid-stream failure"):
        for _ in gen:
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "Mid-stream failure" in span.status.description
    assert span.attributes.get("error.type") == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "Mid-stream failure" in events[0].attributes["exception.message"]


@pytest.mark.asyncio
async def test_async_ollama_streaming_iteration_error_sets_span_status(
    instrument_legacy, tracer_provider, span_exporter
):
    """Same as the sync test but for the async accumulator."""
    from opentelemetry.instrumentation.ollama import _aaccumulate_streaming_response
    from opentelemetry.semconv_ai import LLMRequestTypeValues

    span = tracer_provider.get_tracer(__name__).start_span("ollama.chat")

    async def failing_stream():
        raise Exception("Async mid-stream failure")
        yield  # unreachable, makes this an async generator

    gen = _aaccumulate_streaming_response(
        span, None, None, LLMRequestTypeValues.CHAT, failing_stream()
    )
    with pytest.raises(Exception, match="Async mid-stream failure"):
        async for _ in gen:
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "Async mid-stream failure" in span.status.description
    assert span.attributes.get("error.type") == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "Async mid-stream failure" in events[0].attributes["exception.message"]


def test_ollama_streaming_empty_response_does_not_raise(
    instrument_legacy, tracer_provider, span_exporter
):
    # An empty stream leaves last_response as None; the merge must not TypeError.
    from opentelemetry.instrumentation.ollama import _accumulate_streaming_response
    from opentelemetry.semconv_ai import LLMRequestTypeValues

    span = tracer_provider.get_tracer(__name__).start_span("ollama.chat")

    def empty_stream():
        return
        yield  # unreachable, makes this a generator

    gen = _accumulate_streaming_response(
        span, None, None, LLMRequestTypeValues.CHAT, empty_stream()
    )
    for _ in gen:
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code != StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_ollama_streaming_empty_response_does_not_raise(
    instrument_legacy, tracer_provider, span_exporter
):
    from opentelemetry.instrumentation.ollama import _aaccumulate_streaming_response
    from opentelemetry.semconv_ai import LLMRequestTypeValues

    span = tracer_provider.get_tracer(__name__).start_span("ollama.chat")

    async def empty_stream():
        return
        yield  # unreachable, makes this an async generator

    gen = _aaccumulate_streaming_response(
        span, None, None, LLMRequestTypeValues.CHAT, empty_stream()
    )
    async for _ in gen:
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code != StatusCode.ERROR


def test_ollama_streaming_close_before_reading_ends_span(
    instrument_legacy, tracer_provider, span_exporter
):
    """Closing a stream before reading it must still end the span.

    A generator that has not started never runs its ``finally`` block, so
    the raw accumulation generator leaks the span in this case.
    """
    from opentelemetry.instrumentation.ollama import (
        _StreamingResponse,
        _accumulate_streaming_response,
    )
    from opentelemetry.semconv_ai import LLMRequestTypeValues

    span = tracer_provider.get_tracer(__name__).start_span("ollama.completion")

    class InnerStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield {"response": "hi"}

        def close(self):
            self.closed = True

    inner = InnerStream()
    stream = _StreamingResponse(
        _accumulate_streaming_response(
            span, None, None, LLMRequestTypeValues.COMPLETION, inner
        ),
        span,
        inner,
    )
    stream.close()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "ollama.completion"
    assert inner.closed is True


@pytest.mark.asyncio
async def test_async_ollama_streaming_aclose_closes_inner_stream(
    instrument_legacy, tracer_provider, span_exporter
):
    """aclose() after partial reads must end the span and close the inner stream."""
    from opentelemetry.instrumentation.ollama import (
        _AsyncStreamingResponse,
        _aaccumulate_streaming_response,
    )
    from opentelemetry.semconv_ai import LLMRequestTypeValues

    span = tracer_provider.get_tracer(__name__).start_span("ollama.chat")

    class InnerStream:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield {"message": {"role": "assistant", "content": "hi"}}
            yield {"message": {"role": "assistant", "content": "!"}}

        async def aclose(self):
            self.closed = True

    inner = InnerStream()
    stream = _AsyncStreamingResponse(
        _aaccumulate_streaming_response(
            span, None, None, LLMRequestTypeValues.CHAT, inner
        ),
        span,
        inner,
    )
    await stream.__anext__()
    await stream.aclose()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "ollama.chat"
    assert inner.closed is True
