import os
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

try:
    from anthropic import AsyncAnthropicBedrock
except ImportError:
    AsyncAnthropicBedrock = None


@pytest.fixture
def async_anthropic_bedrock_client(instrument_legacy):
    if AsyncAnthropicBedrock is None:
        pytest.skip("AsyncAnthropicBedrock not available")

    # Try to get credentials from environment first
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "test-key")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "test-secret")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")

    return AsyncAnthropicBedrock(
        aws_region=aws_region,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
    )


# @pytest.mark.skip
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_anthropic_bedrock_with_raw_response(
    instrument_legacy,
    async_anthropic_bedrock_client,
    span_exporter,
    log_exporter,
    reader,
):
    """Test that AsyncAnthropicBedrock with_raw_response.create generates spans"""
    response = await async_anthropic_bedrock_client.messages.with_raw_response.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    # For raw response, content is accessed differently
    response_content = (
        response.parse().content[0].text
        if hasattr(response, "parse")
        else response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] > 0
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )


# @pytest.mark.skip
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_anthropic_bedrock_regular_create(
    instrument_legacy,
    async_anthropic_bedrock_client,
    span_exporter,
    log_exporter,
    reader,
):
    """Test that regular AsyncAnthropicBedrock create works (for comparison)"""
    response = await async_anthropic_bedrock_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] > 0
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )


# @pytest.mark.skip
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_anthropic_bedrock_beta_with_raw_response(
    instrument_legacy,
    async_anthropic_bedrock_client,
    span_exporter,
    log_exporter,
    reader,
):
    """Test that AsyncAnthropicBedrock beta.messages.with_raw_response.create generates spans"""
    response = (
        await async_anthropic_bedrock_client.beta.messages.with_raw_response.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Tell me a joke about OpenTelemetry",
                }
            ],
            model="anthropic.claude-3-haiku-20240307-v1:0",
        )
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    # For raw response, content is accessed differently
    response_content = (
        response.parse().content[0].text
        if hasattr(response, "parse")
        else response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] > 0
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
