"""Chat completion tracing tests.

These use litellm's built-in ``mock_response`` so they need no API key or network —
litellm still resolves the provider (``openai`` for ``gpt-*``) and returns a normalized
``ModelResponse``, exercising the same code path as a real call.
"""

import litellm
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


def test_completion(instrument_legacy, span_exporter):
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        mock_response="The capital of France is Paris.",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]
    span = spans[0]
    attrs = span.attributes

    assert attrs[GenAIAttributes.GEN_AI_SYSTEM] == "openai"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert attrs[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    assert attrs[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == MESSAGES[0]["content"]
    assert (
        attrs[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == "The capital of France is Paris."
    )
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] > 0
    assert response.choices[0].message.content == "The capital of France is Paris."


@pytest.mark.asyncio
async def test_acompletion(instrument_legacy, span_exporter):
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        mock_response="Paris.",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]
    attrs = spans[0].attributes
    assert attrs[GenAIAttributes.GEN_AI_SYSTEM] == "openai"
    assert attrs[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"] == "Paris."


def test_completion_emits_metrics(instrument_legacy, span_exporter, metric_reader):
    from opentelemetry.semconv_ai import Meters

    litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        mock_response="The capital of France is Paris.",
    )

    metrics_data = metric_reader.get_metrics_data()
    metric_names = {
        metric.name
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
    }
    assert Meters.LLM_TOKEN_USAGE in metric_names
    assert Meters.LLM_OPERATION_DURATION in metric_names


def test_completion_no_content_when_disabled(instrument_legacy, span_exporter, monkeypatch):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "False")
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        mock_response="Paris.",
    )

    span = span_exporter.get_finished_spans()[0]
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in span.attributes
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in span.attributes
    # non-content attributes are still present
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
