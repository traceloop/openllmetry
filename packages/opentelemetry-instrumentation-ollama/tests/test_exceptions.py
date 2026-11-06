"""Tests that HTTP errors from the Ollama API are recorded as failed spans."""

import pytest
from unittest.mock import patch
from ollama._client import AsyncClient as OllamaAsyncClient
from ollama._client import Client as OllamaClient
from opentelemetry.instrumentation.ollama import OllamaInstrumentor
from opentelemetry.trace import StatusCode


def test_chat_exception_marks_span_error(span_exporter, tracer_provider, meter_provider):
    # Patch _request on the class BEFORE instrumenting so wrapt wraps the mock.
    # Call chain: ollama.chat → Client._request (wrapt wrapper)
    #   → _dispatch_wrap → _wrap → wrapped() = mock → raises
    #   → our except: set_status(ERROR), record_exception, span.end(), re-raise
    with patch.object(
        OllamaClient, "_request", side_effect=Exception("connection refused")
    ):
        instrumentor = OllamaInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )
        try:
            import ollama

            with pytest.raises(Exception, match="connection refused"):
                ollama.chat(
                    model="llama3",
                    messages=[{"role": "user", "content": "Hello"}],
                )
        finally:
            instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "connection refused" in span.status.description
    assert any(e.name == "exception" for e in span.events)


def test_generate_exception_marks_span_error(span_exporter, tracer_provider, meter_provider):
    with patch.object(
        OllamaClient, "_request", side_effect=Exception("model not found")
    ):
        instrumentor = OllamaInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )
        try:
            import ollama

            with pytest.raises(Exception, match="model not found"):
                ollama.generate(model="llama3", prompt="Hello")
        finally:
            instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "model not found" in span.status.description


@pytest.mark.asyncio
async def test_async_chat_exception_marks_span_error(
    span_exporter, tracer_provider, meter_provider
):
    with patch.object(
        OllamaAsyncClient,
        "_request",
        side_effect=Exception("async connection refused"),
    ):
        instrumentor = OllamaInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )
        try:
            import ollama

            with pytest.raises(Exception, match="async connection refused"):
                await ollama.AsyncClient().chat(
                    model="llama3",
                    messages=[{"role": "user", "content": "Hello"}],
                )
        finally:
            instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "async connection refused" in span.status.description
    assert any(e.name == "exception" for e in span.events)
