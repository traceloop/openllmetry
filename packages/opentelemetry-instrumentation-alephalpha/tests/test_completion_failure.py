"""A failed ``Client.complete`` call must still end its span with ERROR status.

Before the fix the instrumentation only called ``span.end()`` on the happy
path, so an exception raised by the wrapped call left the span unfinished:
with a SimpleSpanProcessor the span was never exported and the trace of the
failed call was lost entirely.
"""

from unittest.mock import patch

import pytest
from aleph_alpha_client import Client, CompletionRequest, Prompt
from opentelemetry.trace import StatusCode

from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor


def test_failed_complete_ends_span_with_error(span_exporter, tracer_provider):
    client = Client(token="test_api_key", host="https://api.aleph-alpha.com")

    # Patch before instrumenting so the instrumentation wraps the mock
    # instead of the mock replacing the instrumentation.
    patcher = patch.object(
        Client, "complete", side_effect=RuntimeError("connection refused")
    )
    patcher.start()
    instrumentor = AlephAlphaInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    try:
        with pytest.raises(RuntimeError, match="connection refused"):
            client.complete(
                CompletionRequest(
                    prompt=Prompt.from_text("hello"), maximum_tokens=10
                ),
                model="luminous-base",
            )
    finally:
        instrumentor.uninstrument()
        patcher.stop()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["alephalpha.completion"]

    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "connection refused" in span.status.description

    exception_events = [
        event for event in span.events if event.name == "exception"
    ]
    assert len(exception_events) == 1
    assert (
        exception_events[0].attributes["exception.type"] == "RuntimeError"
    )
    assert (
        exception_events[0].attributes["exception.message"]
        == "connection refused"
    )
