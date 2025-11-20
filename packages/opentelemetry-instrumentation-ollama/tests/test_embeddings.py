import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_ollama_embeddings_legacy(
    instrument_legacy, ollama_client, span_exporter, log_exporter
):
    ollama_client.embeddings(
        model="llama3", prompt="Tell me a joke about OpenTelemetry"
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.embeddings"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_ollama_embeddings_with_events_with_content(
    instrument_with_content, ollama_client, span_exporter, log_exporter
):
    response = ollama_client.embeddings(
        model="llama3", prompt="Tell me a joke about OpenTelemetry"
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.embeddings"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.get("embedding")},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_ollama_embeddings_with_events_with_no_content(
    instrument_with_no_content, ollama_client, span_exporter, log_exporter
):
    ollama_client.embeddings(
        model="llama3", prompt="Tell me a joke about OpenTelemetry"
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.embeddings"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "ollama"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
