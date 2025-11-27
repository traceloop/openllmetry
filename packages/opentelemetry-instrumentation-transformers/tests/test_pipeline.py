import pytest
import importlib.util
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

# Skip the whole module if torch (or any other heavy backend) is absent.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="`torch` not available – install extras to run these tests",
)


def test_tranformers_pipeline(
    instrument_legacy, span_exporter, log_exporter, transformers_pipeline
):
    prompt_text = "Tell me a joke about OpenTelemetry."
    response = transformers_pipeline(prompt_text)

    spans = span_exporter.get_finished_spans()
    transformers_span = spans[0]
    assert transformers_span.name == "transformers_text_generation_pipeline.call"
    assert transformers_span.attributes.get("gen_ai.system") == "gpt2"
    assert transformers_span.attributes.get("llm.request.type") == "completion"
    assert transformers_span.attributes.get("gen_ai.request.model") == "gpt2"
    assert transformers_span.attributes.get("gen_ai.prompt.0.content") == prompt_text
    assert (
        transformers_span.attributes.get("gen_ai.completion.0.content")
        == response[0]["generated_text"]
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


def test_tranformers_pipeline_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, transformers_pipeline
):
    prompt_text = "Tell me a joke about OpenTelemetry."
    response = transformers_pipeline(prompt_text)

    spans = span_exporter.get_finished_spans()
    transformers_span = spans[0]
    assert transformers_span.name == "transformers_text_generation_pipeline.call"
    assert transformers_span.attributes.get("gen_ai.system") == "gpt2"
    assert transformers_span.attributes.get("llm.request.type") == "completion"
    assert transformers_span.attributes.get("gen_ai.request.model") == "gpt2"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": prompt_text})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response[0]["generated_text"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def test_tranformers_pipeline_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, transformers_pipeline
):
    prompt_text = "Tell me a joke about OpenTelemetry."
    transformers_pipeline(prompt_text)

    spans = span_exporter.get_finished_spans()
    transformers_span = spans[0]
    assert transformers_span.name == "transformers_text_generation_pipeline.call"
    assert transformers_span.attributes.get("gen_ai.system") == "gpt2"
    assert transformers_span.attributes.get("llm.request.type") == "completion"
    assert transformers_span.attributes.get("gen_ai.request.model") == "gpt2"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "transformers"
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
