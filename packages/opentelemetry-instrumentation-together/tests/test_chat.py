import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_together_chat_legacy(
    instrument_legacy, together_client, span_exporter, log_exporter
):
    response = together_client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry."}],
    )

    spans = span_exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "together.chat"
    assert together_span.attributes.get("gen_ai.system") == "TogetherAI"
    assert together_span.attributes.get("llm.request.type") == "chat"
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
        == response.choices[0].message.content
    )
    assert together_span.attributes.get("gen_ai.usage.input_tokens") == 18
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.output_tokens"
    ) + together_span.attributes.get("gen_ai.usage.input_tokens")
    assert together_span.attributes.get("gen_ai.response.id") == "88fa668fac30bb19-MXP"

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_together_chat_with_events_with_content(
    instrument_with_content, together_client, span_exporter, log_exporter
):
    response = together_client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry."}],
    )

    spans = span_exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "together.chat"
    assert together_span.attributes.get("gen_ai.system") == "TogetherAI"
    assert together_span.attributes.get("llm.request.type") == "chat"
    assert (
        together_span.attributes.get("gen_ai.request.model")
        == "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    assert together_span.attributes.get("gen_ai.usage.input_tokens") == 18
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.output_tokens"
    ) + together_span.attributes.get("gen_ai.usage.input_tokens")
    assert together_span.attributes.get("gen_ai.response.id") == "88fa668fac30bb19-MXP"

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
        "finish_reason": "eos",
        "message": {"content": response.choices[0].message.content},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_together_chat_with_events_with_no_content(
    instrument_with_no_content, together_client, span_exporter, log_exporter
):
    together_client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry."}],
    )

    spans = span_exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "together.chat"
    assert together_span.attributes.get("gen_ai.system") == "TogetherAI"
    assert together_span.attributes.get("llm.request.type") == "chat"
    assert (
        together_span.attributes.get("gen_ai.request.model")
        == "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    assert together_span.attributes.get("gen_ai.usage.input_tokens") == 18
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.output_tokens"
    ) + together_span.attributes.get("gen_ai.usage.input_tokens")
    assert together_span.attributes.get("gen_ai.response.id") == "88fa668fac30bb19-MXP"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "eos",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "together"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
