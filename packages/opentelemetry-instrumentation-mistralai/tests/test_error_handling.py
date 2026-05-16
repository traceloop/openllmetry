from unittest.mock import patch

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


def test_mistralai_streaming_iteration_error_sets_span_status(
    instrument_legacy, tracer_provider, span_exporter
):
    """Errors raised during stream iteration must still finalize the span
    with ERROR status, error.type, and an exception event."""
    from opentelemetry.instrumentation.mistralai import _accumulate_streaming_response
    from opentelemetry.semconv_ai import LLMRequestTypeValues

    span = tracer_provider.get_tracer(__name__).start_span("mistralai.chat")

    def failing_stream():
        raise Exception("Mid-stream failure")
        yield  # unreachable, makes this a generator

    gen = _accumulate_streaming_response(
        span, None, LLMRequestTypeValues.CHAT, failing_stream()
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
