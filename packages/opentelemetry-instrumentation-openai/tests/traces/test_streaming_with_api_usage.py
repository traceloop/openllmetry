import pytest
from openai import OpenAI
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.semconv_ai import SpanAttributes


@pytest.fixture
def api_usage_provider_client():
    """Client for testing API providers that include usage information in streaming responses, use deepseek here"""
    return OpenAI(
        api_key="test-api-key",
        base_url="https://api.deepseek.com/beta"
    )


@pytest.mark.vcr
def test_streaming_with_api_usage_capture(
    instrument_legacy, span_exporter, api_usage_provider_client
):
    """Test that streaming responses with API usage information are properly captured"""
    response = api_usage_provider_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    response_content = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            response_content += chunk.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "openai.chat"

    # Check that token usage is captured from API response
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0

    # Verify that the response content is meaningful
    assert len(response_content) > 0
    assert span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL) == "deepseek-chat"


@pytest.mark.vcr
def test_streaming_with_api_usage_and_events(
    instrument_with_content, span_exporter, log_exporter, api_usage_provider_client
):
    """Test that streaming responses with API usage work with event logging"""
    response = api_usage_provider_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "What is OpenTelemetry?"}],
        stream=True,
    )

    for chunk in response:
        pass  # Just consume the stream

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]

    # Check that usage metrics are captured from API response
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0

    # Check event logs
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2  # At least user message and assistant choice
