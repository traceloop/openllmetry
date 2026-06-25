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


def test_streaming_preserves_custom_stream_wrapper(instrument_legacy, span_exporter):
    """The wrapped stream must remain a litellm CustomStreamWrapper so callers that
    rely on its helper methods / isinstance checks keep working."""
    stream = litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        stream=True,
        mock_response="one two three",
    )
    assert isinstance(stream, litellm.CustomStreamWrapper)
    list(stream)


def test_streaming_span_ends_on_early_break(instrument_legacy, span_exporter):
    """Abandoning the stream early must still end the span exactly once — a bare
    generator would leave it open forever (regression for span leak on partial consumption)."""
    stream = litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        stream=True,
        mock_response="one two three four five",
    )
    for chunk in stream:
        break
    stream.close()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]


def test_streaming_span_ends_on_consumer_error(instrument_legacy, span_exporter):
    """An exception in the consumer loop must not leak the span; closing the stream ends it."""
    stream = litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        stream=True,
        mock_response="one two three",
    )
    with pytest.raises(RuntimeError):
        for _ in stream:
            raise RuntimeError("consumer blew up")
    stream.close()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]


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
