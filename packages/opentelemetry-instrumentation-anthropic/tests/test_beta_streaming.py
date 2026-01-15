"""Tests for beta streaming API.

These tests verify that the OpenTelemetry instrumentation works correctly with
the beta streaming API (client.beta.messages.stream()), which was the subject
of a fix that moved AsyncMessages.stream from WRAPPED_AMETHODS to WRAPPED_METHODS.

The beta streaming API returns an async context manager, not a coroutine, so it
needs the sync wrapper (not the async wrapper that would await it).

Related:
- Issue #3178: https://github.com/traceloop/openllmetry/issues/3178
- PR #3220: https://github.com/traceloop/openllmetry/pull/3220 (fixed non-beta)
"""

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_anthropic_beta_message_stream_manager_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    """Test sync beta streaming with legacy attributes."""
    response_content = ""
    with anthropic_client.beta.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a very short joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a very short joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] >= 1
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_beta_message_stream_manager_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter, reader
):
    """Test async beta streaming with legacy attributes.

    This is the main test case for the fix. Before the fix, this would fail with:
        RuntimeWarning: coroutine '_awrap' was never awaited
        TypeError: 'coroutine' object does not support the asynchronous context manager protocol

    The fix moves beta.messages.AsyncMessages.stream from WRAPPED_AMETHODS to
    WRAPPED_METHODS, using the sync wrapper instead of async wrapper.
    """
    response_content = ""
    async with async_anthropic_client.beta.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a very short joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a very short joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] >= 1
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_beta_message_stream_manager_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter, reader
):
    """Test async beta streaming with events and content logging enabled."""
    response_content = ""
    async with async_anthropic_client.beta.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a very short joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] >= 1
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_beta_message_stream_manager_with_events_with_no_content(
    instrument_with_no_content,
    async_anthropic_client,
    span_exporter,
    log_exporter,
    reader,
):
    """Test async beta streaming with events but content logging disabled."""
    response_content = ""
    async with async_anthropic_client.beta.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a very short joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] >= 1
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2
