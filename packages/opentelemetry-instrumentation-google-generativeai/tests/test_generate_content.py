import pytest
from unittest.mock import MagicMock
from opentelemetry.instrumentation.google_generativeai import (
    GoogleGenerativeAiInstrumentor,
)
from opentelemetry.trace import StatusCode, SpanKind
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.sdk._logs import LogData

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "gemini"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


@pytest.fixture
def mock_instrumentor():
    instrumentor = GoogleGenerativeAiInstrumentor()
    instrumentor.instrument = MagicMock()
    instrumentor.uninstrument = MagicMock()
    return instrumentor


@pytest.mark.vcr
def test_client_spans(exporter, genai_client):
    genai_client.chats.create(model="gemini-2.5-flash").send_message("What is ai?")
    spans = exporter.get_finished_spans()

    assert len(spans) > 0, "No spans were recorded"

    span = next(
        (s for s in spans if s.name == "gemini.generate_content"),
        None,
    )
    assert span is not None, "gemini.generate_content span not found"

    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK

    attrs = span.attributes

    assert attrs[SpanAttributes.LLM_SYSTEM] == "Google"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gemini-2.5-flash"
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gemini-2.5-flash"

    assert "gen_ai.prompt.0.content" in attrs
    assert attrs["gen_ai.prompt.0.role"] == "user"

    assert "gen_ai.completion.0.content" in attrs
    assert attrs["gen_ai.completion.0.role"] == "assistant"

    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] > 0
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] > 0
