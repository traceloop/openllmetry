
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
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "gemini"

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
        (s for s in spans if s.name == "gemini.generate_content"),
        None,
    )
    assert span is not None, "gemini.generate_content span not found"

    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK

    attrs = span.attributes

    assert attrs[GenAIAttributes.GEN_AI_SYSTEM] == "Google"
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


@pytest.mark.vcr
def test_generate_metrics(metrics_test_context, genai_client):
    _, reader = metrics_test_context

    # ---- Trigger a generic GenAI request ----
    genai_client.chats.create(model="gemini-2.5-flash").send_message("What is ai?")

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    # ---- ResourceMetrics ----
    assert resource_metrics, "No ResourceMetrics emitted"

    rm = resource_metrics[0]
    assert rm.scope_metrics, "No ScopeMetrics found"

    scope_metrics = rm.scope_metrics[0]

    # ---- Instrumentation scope (generic check) ----
    scope = scope_metrics.scope
    assert scope.name, "Instrumentation scope name is missing"

    metrics = {m.name: m for m in scope_metrics.metrics}

    # ---- Required metrics (semantic conventions) ----
    required_metrics = {
        Meters.LLM_OPERATION_DURATION,
        Meters.LLM_TOKEN_USAGE,
    }
    assert required_metrics.issubset(metrics.keys())

    duration_metric = metrics[Meters.LLM_OPERATION_DURATION]

    assert duration_metric.unit is not None
    assert duration_metric.data.data_points

    duration_dp = duration_metric.data.data_points[0]

    # Minimal semantic validation
    assert duration_dp.count >= 1
    assert duration_dp.sum >= 0

    # Required attributes (values are intentionally not hard-coded)
    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in duration_dp.attributes
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
        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in dp.attributes
