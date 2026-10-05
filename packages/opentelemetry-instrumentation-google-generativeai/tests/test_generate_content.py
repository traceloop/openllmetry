import json
from unittest.mock import MagicMock

import pytest
from opentelemetry.trace import StatusCode, SpanKind
from opentelemetry.semconv_ai import (
    SpanAttributes,
    Meters
)
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(
        GenAIAttributes.GEN_AI_PROVIDER_NAME
    ) == GenAIAttributes.GenAiProviderNameValues.GCP_GEN_AI.value

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


@pytest.mark.vcr
def test_client_spans(exporter, genai_client):
    genai_client.chats.create(model="gemini-2.5-flash").send_message("What is ai?")
    spans = exporter.get_finished_spans()

    assert len(spans) > 0, "No spans were recorded"

    span = next(
        (s for s in spans if s.name.startswith("generate_content ")),
        None,
    )
    assert span is not None, "generate_content span not found"

    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK

    attrs = span.attributes

    assert (
        attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.GCP_GEN_AI.value
    )
    assert (
        attrs[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.GENERATE_CONTENT.value
    )
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gemini-2.5-flash"
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gemini-2.5-flash"

    # Input messages are now a single JSON array
    assert "gen_ai.input.messages" in attrs
    input_messages = json.loads(attrs["gen_ai.input.messages"])
    assert len(input_messages) > 0
    assert input_messages[0]["role"] == "user"
    assert "parts" in input_messages[0]
    assert input_messages[0]["parts"][0]["type"] == "text"
    assert "content" in input_messages[0]["parts"][0]

    # Output messages are now a single JSON array (parts-based OTel schema)
    assert "gen_ai.output.messages" in attrs
    output_messages = json.loads(attrs["gen_ai.output.messages"])
    assert len(output_messages) > 0
    assert output_messages[0]["role"] == "assistant"
    assert "parts" in output_messages[0]
    assert output_messages[0]["parts"][0]["type"] == "text"
    assert "content" in output_messages[0]["parts"][0]

    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in attrs
    finish_reasons = attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
    assert isinstance(finish_reasons, (list, tuple))
    assert len(finish_reasons) > 0

    assert attrs[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] > 0
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] > 0


@pytest.mark.vcr
def test_generate_metrics(metrics_test_context, genai_client, exporter):
    """`exporter` fixture instruments with the session MeterProvider so metrics are recorded."""
    _, reader = metrics_test_context

    # ---- Trigger a generic GenAI request ----
    genai_client.chats.create(model="gemini-2.5-flash").send_message("What is ai?")

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    # ---- ResourceMetrics ----
    assert resource_metrics, "No ResourceMetrics emitted"

    rm = resource_metrics[0]
    assert rm.scope_metrics, "No ScopeMetrics found"

    # Aggregate across scopes: SDK internal metrics share the reader with app instrumentation.
    metrics = {}
    instrument_scope = None
    for sm in rm.scope_metrics:
        for m in sm.metrics:
            metrics[m.name] = m
        if any(
            name in (Meters.LLM_OPERATION_DURATION, Meters.LLM_TOKEN_USAGE)
            for name in (x.name for x in sm.metrics)
        ):
            instrument_scope = sm.scope

    assert instrument_scope and instrument_scope.name, "GenAI metrics scope not found"

    # ---- Required metrics (semantic conventions) ----
    required_metrics = {
        Meters.LLM_OPERATION_DURATION,
        Meters.LLM_TOKEN_USAGE,
    }
    assert required_metrics.issubset(metrics.keys())

    duration_metric = metrics[Meters.LLM_OPERATION_DURATION]

    assert duration_metric.unit is not None
    assert duration_metric.data.data_points

    duration_dp = next(
        dp for dp in duration_metric.data.data_points if dp.count >= 1
    )

    # Minimal semantic validation
    assert duration_dp.count >= 1
    assert duration_dp.sum >= 0

    # Required attributes (values are intentionally not hard-coded)
    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in duration_dp.attributes
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in duration_dp.attributes
    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in duration_dp.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in duration_dp.attributes

    token_metric = metrics[Meters.LLM_TOKEN_USAGE]

    assert token_metric.unit == "token"
    assert token_metric.data.data_points

    token_points_by_type = {
        dp.attributes.get(GenAIAttributes.GEN_AI_TOKEN_TYPE): dp
        for dp in token_metric.data.data_points
    }

    # Both input & output tokens must exist
    assert {"input", "output"}.issubset(token_points_by_type.keys())

    for token_type, dp in token_points_by_type.items():
        assert dp.count >= 1
        assert dp.sum >= 0

        # Required semantic attributes
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in dp.attributes
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in dp.attributes
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in dp.attributes
        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in dp.attributes


def test_set_model_request_attributes_reads_system_instruction_from_config(monkeypatch):
    from opentelemetry.instrumentation.google_generativeai import span_utils as su

    monkeypatch.setattr(su, "should_send_prompts", lambda: True)

    span = MagicMock()
    span.is_recording.return_value = True

    class GenerateContentConfig:
        system_instruction = "Reply in one sentence."

    su.set_model_request_attributes(span, {"config": GenerateContentConfig()}, "gemini-pro")

    sys_attr_calls = [
        c[0]
        for c in span.set_attribute.call_args_list
        if c[0][0] == GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS
    ]
    assert len(sys_attr_calls) == 1
    parts = json.loads(sys_attr_calls[0][1])
    assert parts[0]["type"] == "text"
    assert parts[0]["content"] == "Reply in one sentence."


def test_set_model_response_attributes_emits_cache_read_tokens():
    from opentelemetry.instrumentation.google_generativeai import span_utils as su

    span = MagicMock()
    span.is_recording.return_value = True

    class UsageMetadata:
        total_token_count = 150
        candidates_token_count = 50
        prompt_token_count = 100
        cached_content_token_count = 40

    class Response:
        usage_metadata = UsageMetadata()
        response_id = None

    su.set_model_response_attributes(span, Response(), "gemini-pro", token_histogram=None)

    set_attr_calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert set_attr_calls[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 40


def test_set_model_response_attributes_no_cache_read_tokens_when_absent():
    from opentelemetry.instrumentation.google_generativeai import span_utils as su

    span = MagicMock()
    span.is_recording.return_value = True

    class UsageMetadata:
        total_token_count = 150
        candidates_token_count = 50
        prompt_token_count = 100
        # no cached_content_token_count attribute

    class Response:
        usage_metadata = UsageMetadata()
        response_id = None

    su.set_model_response_attributes(span, Response(), "gemini-pro", token_histogram=None)

    set_attr_keys = [c[0][0] for c in span.set_attribute.call_args_list]
    assert SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in set_attr_keys


@pytest.mark.parametrize("as_dict", [True, False])
def test_set_model_request_attributes_reads_parameters_and_system_instruction_from_config(
    monkeypatch, as_dict
):
    from google.genai import types
    from opentelemetry.instrumentation.google_generativeai import span_utils as su

    monkeypatch.setattr(su, "should_send_prompts", lambda: True)

    span = MagicMock()
    span.is_recording.return_value = True

    settings = dict(
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_output_tokens=50,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        seed=7,
        stop_sequences=["END"],
        system_instruction="be brief",
        tools=[{"function_declarations": [{"name": "get_weather"}]}],
    )
    config = dict(settings) if as_dict else types.GenerateContentConfig(**settings)

    su.set_model_request_attributes(span, {"config": config}, "gemini-2.5-flash")

    attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 50
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY] == 0.1
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY] == 0.2
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_SEED] == 7
    assert list(attrs[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES]) == ["END"]

    sys_parts = json.loads(attrs[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
    assert sys_parts == [{"type": "text", "content": "be brief"}]

    tool_defs = json.loads(attrs[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
    assert len(tool_defs) == 1

