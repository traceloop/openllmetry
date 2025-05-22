import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_cohere_chat_legacy(
    span_exporter, log_exporter, instrument_legacy, cohere_client
):
    res = cohere_client.chat(model="command", message="Tell me a joke, pirate style")

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke, pirate style"
    )
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == res.text
    )
    assert cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 58
    assert cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "440f51f4-3e47-44b6-a5d7-5ba33edcfc58"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_chat_with_events_with_content(
    span_exporter, log_exporter, instrument_with_content, cohere_client
):
    user_message = "Tell me a joke, pirate style"
    res = cohere_client.chat(model="command", message=user_message)

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": user_message})

    # Validate model response Event
    choice_event = {
        "index": 0,
        "finish_reason": "COMPLETE",
        "message": {"content": res.text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_cohere_chat_with_events_with_no_content(
    span_exporter, log_exporter, instrument_with_no_content, cohere_client
):
    user_message = "Tell me a joke, pirate style"
    cohere_client.chat(model="command", message=user_message)

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate model response Event
    choice_event = {
        "index": 0,
        "finish_reason": "COMPLETE",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.COHERE.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
