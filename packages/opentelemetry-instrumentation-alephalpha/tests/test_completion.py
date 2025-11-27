import pytest
from aleph_alpha_client import CompletionRequest, Prompt
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_alephalpha_completion(
    span_exporter, log_exporter, aleph_alpha_client, instrument_legacy
):
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
    }
    request = CompletionRequest(**params)
    response = aleph_alpha_client.complete(request, model="luminous-base")

    spans = span_exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "alephalpha.completion"
    assert together_span.attributes.get("gen_ai.system") == "AlephAlpha"
    assert together_span.attributes.get("llm.request.type") == "completion"
    assert together_span.attributes.get("gen_ai.request.model") == "luminous-base"
    assert (
        together_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry."
    )
    assert (
        together_span.attributes.get("gen_ai.completion.0.content")
        == response.completions[0].completion
    )
    assert together_span.attributes.get("gen_ai.usage.input_tokens") == 9
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.output_tokens"
    ) + together_span.attributes.get("gen_ai.usage.input_tokens")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_alephalpha_completion_with_events_with_content(
    span_exporter, log_exporter, aleph_alpha_client, instrument_with_content
):
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
    }
    request = CompletionRequest(**params)
    response = aleph_alpha_client.complete(request, model="luminous-base")

    spans = span_exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "alephalpha.completion"
    assert together_span.attributes.get("gen_ai.system") == "AlephAlpha"
    assert together_span.attributes.get("llm.request.type") == "completion"
    assert together_span.attributes.get("gen_ai.request.model") == "luminous-base"
    assert together_span.attributes.get("gen_ai.usage.input_tokens") == 9
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.output_tokens"
    ) + together_span.attributes.get("gen_ai.usage.input_tokens")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {
        "content": [
            {
                "controls": [],
                "data": "Tell me a joke about OpenTelemetry.",
                "type": "text",
            }
        ]
    }
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "maximum_tokens",
        "message": {"content": response.completions[0].completion},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_alephalpha_completion_with_events_with_no_content(
    span_exporter, log_exporter, aleph_alpha_client, instrument_with_no_content
):
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
    }
    request = CompletionRequest(**params)
    aleph_alpha_client.complete(request, model="luminous-base")

    spans = span_exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "alephalpha.completion"
    assert together_span.attributes.get("gen_ai.system") == "AlephAlpha"
    assert together_span.attributes.get("llm.request.type") == "completion"
    assert together_span.attributes.get("gen_ai.request.model") == "luminous-base"
    assert together_span.attributes.get("gen_ai.usage.input_tokens") == 9
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.output_tokens"
    ) + together_span.attributes.get("gen_ai.usage.input_tokens")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "maximum_tokens",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "alephalpha"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
