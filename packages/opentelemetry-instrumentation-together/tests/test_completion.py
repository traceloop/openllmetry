import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_together_completion_legacy(
    instrument_legacy, together_client, span_exporter, log_exporter
):
    response = together_client.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt="Tell me a joke about OpenTelemetry.",
    )

    spans = span_exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "together.completion"
    assert together_span.attributes.get("gen_ai.system") == "TogetherAI"
    assert together_span.attributes.get("llm.request.type") == "completion"
    assert (
        together_span.attributes.get("gen_ai.request.model")
        == "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    assert (
        together_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry."
    )
    assert (
        together_span.attributes.get("gen_ai.completion.0.content")
        == response.choices[0].text
    )
    assert together_span.attributes.get("gen_ai.usage.prompt_tokens") == 10
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + together_span.attributes.get("gen_ai.usage.prompt_tokens")
    assert together_span.attributes.get("gen_ai.response.id") == "88fa66988e400e83-MXP"

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_together_completion_with_events_with_content(
    instrument_with_content, together_client, span_exporter, log_exporter
):
    response = together_client.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt="Tell me a joke about OpenTelemetry.",
    )

    spans = span_exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "together.completion"
    assert together_span.attributes.get("gen_ai.system") == "TogetherAI"
    assert together_span.attributes.get("llm.request.type") == "completion"
    assert (
        together_span.attributes.get("gen_ai.request.model")
        == "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    assert together_span.attributes.get("gen_ai.usage.prompt_tokens") == 10
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + together_span.attributes.get("gen_ai.usage.prompt_tokens")
    assert together_span.attributes.get("gen_ai.response.id") == "88fa66988e400e83-MXP"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry."},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {"content": response.choices[0].text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_together_completion_with_events_with_no_content(
    instrument_with_no_content, together_client, span_exporter, log_exporter
):
    together_client.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt="Tell me a joke about OpenTelemetry.",
    )

    spans = span_exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "together.completion"
    assert together_span.attributes.get("gen_ai.system") == "TogetherAI"
    assert together_span.attributes.get("llm.request.type") == "completion"
    assert (
        together_span.attributes.get("gen_ai.request.model")
        == "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    assert together_span.attributes.get("gen_ai.usage.prompt_tokens") == 10
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + together_span.attributes.get("gen_ai.usage.prompt_tokens")
    assert together_span.attributes.get("gen_ai.response.id") == "88fa66988e400e83-MXP"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "together"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
