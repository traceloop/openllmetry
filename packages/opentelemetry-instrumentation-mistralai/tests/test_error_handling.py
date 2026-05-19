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
