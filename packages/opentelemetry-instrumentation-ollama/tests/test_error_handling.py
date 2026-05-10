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
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
