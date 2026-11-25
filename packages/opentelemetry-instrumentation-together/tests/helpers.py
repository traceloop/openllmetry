"""Assertions shared by the chat and completion tests."""

import json
from enum import Enum

from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import GenAISystem, SpanAttributes

PROVIDER = GenAISystem.TOGETHER_AI.value
MODEL = "openai/gpt-oss-120b"
PROMPT = "Tell me a joke about OpenTelemetry."

# Legacy flat attributes this instrumentation no longer emits.
LEGACY_PREFIXES = ("gen_ai.prompt.", "gen_ai.completion.", "llm.")


def raw_value(value):
    return value.value if isinstance(value, Enum) else value


def otel_finish_reason(choice):
    """Together reports "eos" for a natural stop; OTel calls that "stop"."""
    reason = raw_value(choice.finish_reason)
    return {"eos": "stop", "tool_calls": "tool_call"}.get(reason, reason)


def get_single_span(span_exporter, operation):
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"{operation} {MODEL}"]
    return spans[0]


def assert_common_span_attributes(span, operation, response):
    attributes = span.attributes
    assert attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == PROVIDER
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == operation
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == MODEL
    assert attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == response.model
    assert attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response.id
    assert attributes[SpanAttributes.GEN_AI_IS_STREAMING] is False
    assert attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == tuple(
        otel_finish_reason(choice) for choice in response.choices
    )
    assert attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == response.usage.prompt_tokens
    assert attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == response.usage.completion_tokens
    assert attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == response.usage.total_tokens
    assert not [key for key in attributes if key.startswith(LEGACY_PREFIXES)]
    assert GenAIAttributes.GEN_AI_SYSTEM not in attributes


def assert_no_content_attributes(span):
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def input_messages(span):
    return json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])


def output_messages(span):
    return json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == PROVIDER

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


def _data_points(reader, name):
    metrics_data = reader.get_metrics_data()
    if metrics_data is None:
        return []
    return [
        point
        for resource_metrics in metrics_data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def assert_metrics(reader, operation, response):
    expected = {
        GenAIAttributes.GEN_AI_PROVIDER_NAME: PROVIDER,
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: MODEL,
        GenAIAttributes.GEN_AI_RESPONSE_MODEL: response.model,
    }

    token_points = {
        point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]: point
        for point in _data_points(reader, GenAIMetrics.GEN_AI_CLIENT_TOKEN_USAGE)
    }
    assert set(token_points) == {"input", "output"}
    assert token_points["input"].sum == response.usage.prompt_tokens
    assert token_points["output"].sum == response.usage.completion_tokens
    for token_type, point in token_points.items():
        assert dict(point.attributes) == {**expected, GenAIAttributes.GEN_AI_TOKEN_TYPE: token_type}

    duration_points = _data_points(reader, GenAIMetrics.GEN_AI_CLIENT_OPERATION_DURATION)
    assert len(duration_points) == 1
    assert dict(duration_points[0].attributes) == expected
    assert duration_points[0].count == 1
    assert duration_points[0].sum > 0


def assert_error_duration_metric(reader, operation, error_type):
    duration_points = _data_points(reader, GenAIMetrics.GEN_AI_CLIENT_OPERATION_DURATION)
    assert len(duration_points) == 1
    assert dict(duration_points[0].attributes) == {
        GenAIAttributes.GEN_AI_PROVIDER_NAME: PROVIDER,
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: MODEL,
        ERROR_TYPE: error_type,
    }
    assert not _data_points(reader, GenAIMetrics.GEN_AI_CLIENT_TOKEN_USAGE)
