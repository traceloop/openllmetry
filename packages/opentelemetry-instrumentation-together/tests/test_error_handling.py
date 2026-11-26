from unittest.mock import patch

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace.status import StatusCode

from .helpers import MODEL, PROVIDER, assert_error_duration_metric

CHAT = GenAIAttributes.GenAiOperationNameValues.CHAT.value


def test_together_error_sets_span_status(instrument_legacy, caller, span_exporter, reader):
    with patch(caller.request_patch_target(), side_effect=Exception("API connection error")):
        with pytest.raises(Exception, match="API connection error"):
            caller.chat(
                model=MODEL,
                messages=[{"role": "user", "content": "Tell me a joke"}],
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == f"{CHAT} {MODEL}"
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == PROVIDER
    assert span.status.status_code == StatusCode.ERROR
    assert "API connection error" in span.status.description
    assert span.attributes.get(ERROR_TYPE) == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "API connection error" in events[0].attributes["exception.message"]
    assert_error_duration_metric(reader, CHAT, "Exception")
