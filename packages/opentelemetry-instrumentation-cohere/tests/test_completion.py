import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_cohere_completion_legacy(
    span_exporter, log_exporter, instrument_legacy, cohere_client
):
    res = cohere_client.generate(model="command", prompt="Tell me a joke, pirate style")

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.completion"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "completion"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == res.generations[0].text
    )
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "64c671fc-c536-41fc-adbd-5f7c81177371"
    )
    assert (
        cohere_span.attributes.get("gen_ai.response.0.id")
        == "13255d0a-eef8-47fc-91f7-d2607d228fbf"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_completion_with_events_with_content(
    span_exporter, log_exporter, instrument_with_content, cohere_client
):
    user_message = "Tell me a joke, pirate style"
    res = cohere_client.generate(model="command", prompt=user_message)

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.completion"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "completion"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "command"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": user_message})

    # Validate model response Event
    choice_event = {
        "index": 0,
        "finish_reason": "COMPLETE",
        "message": {"content": res.generations[0].text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_cohere_completion_with_events_with_no_content(
    span_exporter, log_exporter, instrument_with_no_content, cohere_client
):
    cohere_client.generate(model="command", prompt="Tell me a joke, pirate style")

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.completion"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "completion"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "command"

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
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.COHERE.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
