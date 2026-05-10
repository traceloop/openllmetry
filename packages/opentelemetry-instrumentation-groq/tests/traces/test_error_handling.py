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
