from unittest.mock import patch

import pytest
from opentelemetry.trace.status import StatusCode


def test_together_error_sets_span_status(instrument_legacy, caller, span_exporter):
    with patch(caller.request_patch_target(), side_effect=Exception("API connection error")):
        with pytest.raises(Exception, match="API connection error"):
            caller.chat(
                model="meta-llama/Llama-3-8b-chat-hf",
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
