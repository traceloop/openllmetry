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
