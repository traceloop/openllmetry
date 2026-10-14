import logging

import cohere
import pytest
from opentelemetry.instrumentation.cohere import CohereInstrumentor
from opentelemetry.trace import StatusCode


@pytest.mark.asyncio
async def test_async_cohere_exception_records_error_once(
    caplog, monkeypatch, span_exporter, tracer_provider
):
    async def failing_chat(self, *args, **kwargs):
        raise RuntimeError("cohere async failure")

    monkeypatch.setattr(cohere.AsyncClient, "chat", failing_chat)
    instrumentor = CohereInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    try:
        with caplog.at_level(logging.WARNING, logger="opentelemetry.sdk.trace"):
            with pytest.raises(RuntimeError, match="cohere async failure"):
                await cohere.AsyncClient("test_api_key").chat(
                    model="command", message="Tell me a joke, pirate style"
                )
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "cohere.chat"
    assert span.status.status_code == StatusCode.ERROR
    assert "cohere async failure" in span.status.description
    assert any(
        event.name == "exception"
        and "cohere async failure" in event.attributes.get("exception.message", "")
        for event in span.events
    )
    assert not any(
        "Calling end() on an ended span" in record.message for record in caplog.records
    )
