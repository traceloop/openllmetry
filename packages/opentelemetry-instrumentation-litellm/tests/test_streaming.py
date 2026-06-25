"""Streaming tracing tests (offline via litellm mock_response)."""

import litellm
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

MESSAGES = [{"role": "user", "content": "Count to three."}]


def test_streaming_completion(instrument_legacy, span_exporter):
    stream = litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        stream=True,
        mock_response="one two three",
    )

    chunks = list(stream)
    assert len(chunks) > 0

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]
    attrs = spans[0].attributes
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert attrs[GenAIAttributes.GEN_AI_SYSTEM] == "openai"
    assert attrs[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"] == "one two three"


@pytest.mark.asyncio
async def test_astreaming_completion(instrument_legacy, span_exporter):
    stream = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        stream=True,
        mock_response="one two three",
    )

    chunks = [chunk async for chunk in stream]
    assert len(chunks) > 0

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]
    attrs = spans[0].attributes
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert attrs[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"] == "one two three"
