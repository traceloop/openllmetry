from unittest.mock import patch

import pytest
from opentelemetry.trace.status import StatusCode


def test_groq_error_sets_span_status(instrument_legacy, groq_client, span_exporter):
    with patch("httpx.Client.send", side_effect=Exception("API connection error")):
        with pytest.raises(Exception):
            groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Tell me a joke"}],
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") is not None
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1


@pytest.mark.asyncio
async def test_async_groq_error_sets_span_status(instrument_legacy, async_groq_client, span_exporter):
    with patch("httpx.AsyncClient.send", side_effect=Exception("Async API error")):
        with pytest.raises(Exception):
            await async_groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Tell me a joke"}],
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") is not None
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1


def test_groq_streaming_iteration_error_sets_span_status(
    instrument_legacy, tracer_provider, span_exporter
):
    """Errors raised during stream iteration must still finalize the span
    with ERROR status, error.type, and an exception event."""
    from opentelemetry.instrumentation.groq import _create_stream_processor

    span = tracer_provider.get_tracer(__name__).start_span("groq.chat")

    def failing_stream():
        raise Exception("Mid-stream failure")
        yield  # unreachable, makes this a generator

    gen = _create_stream_processor(failing_stream(), span, None)
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
async def test_async_groq_streaming_iteration_error_sets_span_status(
    instrument_legacy, tracer_provider, span_exporter
):
    """Same as the sync test but for the async stream processor."""
    from opentelemetry.instrumentation.groq import _create_async_stream_processor

    span = tracer_provider.get_tracer(__name__).start_span("groq.chat")

    async def failing_stream():
        raise Exception("Async mid-stream failure")
        yield  # unreachable, makes this an async generator

    gen = _create_async_stream_processor(failing_stream(), span, None)
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
