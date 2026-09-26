"""Unit tests for reasoning-token usage extraction (gen_ai.usage.reasoning_tokens).

Covers the three paths that consume Anthropic `usage.output_tokens_details`:
the sync and async non-streaming `_set_token_usage` helpers in `__init__.py`,
and the streaming path in `streaming.py`.
"""

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.anthropic import _aset_token_usage, _set_token_usage
from opentelemetry.instrumentation.anthropic.streaming import (
    _process_response_item,
    _set_token_usage as _set_streaming_token_usage,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

REASONING_TOKENS = SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS


def _make_usage(**overrides):
    """Build a stub Anthropic usage object, overriding any supplied fields."""
    usage = {
        "input_tokens": 10,
        "output_tokens": 25,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens_details": SimpleNamespace(reasoning_tokens=15),
    }
    usage.update(overrides)
    return SimpleNamespace(**usage)


def _make_response(usage):
    """Build a stub Anthropic message response object around *usage*."""
    return SimpleNamespace(
        usage=usage,
        content=[SimpleNamespace(type="text", text="hi")],
        stop_reason="end_turn",
        model="claude-3-7-sonnet-20250219",
    )


def _finished_span_attributes(span_exporter):
    """Return the attributes of the first finished span exported by *span_exporter*."""
    return dict(span_exporter.get_finished_spans()[0].attributes)


@pytest.fixture
def tracer(tracer_provider):
    """Provide a named tracer backed by the *tracer_provider* fixture."""
    return tracer_provider.get_tracer("test-reasoning-usage")


def test_set_token_usage_emits_reasoning_tokens(tracer, span_exporter):
    """Sync non-streaming responses must emit gen_ai.usage.reasoning_tokens."""
    with tracer.start_as_current_span("test") as span:
        _set_token_usage(span, None, {}, _make_response(_make_usage()))

    attributes = _finished_span_attributes(span_exporter)
    assert attributes[REASONING_TOKENS] == 15
    assert attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 25


@pytest.mark.asyncio
async def test_aset_token_usage_emits_reasoning_tokens(tracer, span_exporter):
    """Async non-streaming responses must emit gen_ai.usage.reasoning_tokens."""
    with tracer.start_as_current_span("test") as span:
        await _aset_token_usage(span, None, {}, _make_response(_make_usage()))

    attributes = _finished_span_attributes(span_exporter)
    assert attributes[REASONING_TOKENS] == 15


def test_set_token_usage_omits_reasoning_when_details_absent(tracer, span_exporter):
    """Reasoning tokens must be omitted when output_tokens_details is absent."""
    with tracer.start_as_current_span("test") as span:
        _set_token_usage(span, None, {}, _make_response(_make_usage(output_tokens_details=None)))

    attributes = _finished_span_attributes(span_exporter)
    assert REASONING_TOKENS not in attributes


def test_process_response_item_keeps_latest_streaming_reasoning_tokens():
    """Streaming must retain the latest cumulative reasoning-token count, not a sum."""
    complete_response = {"events": [], "model": "", "usage": {}, "id": ""}

    _process_response_item(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                model="claude-3-7-sonnet-20250219",
                id="msg_1",
                usage={
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            ),
        ),
        complete_response,
    )
    _process_response_item(
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage={
                "output_tokens": 16,
                "output_tokens_details": {"reasoning_tokens": 15},
            },
        ),
        complete_response,
    )
    _process_response_item(
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage={
                "output_tokens": 25,
                "output_tokens_details": {"reasoning_tokens": 27},
            },
        ),
        complete_response,
    )

    usage = complete_response["usage"]
    assert usage["output_tokens_details"]["reasoning_tokens"] == 27


def test_streaming_set_token_usage_emits_reasoning_tokens(tracer, span_exporter):
    """Streaming span attributes must include the accumulated reasoning-token count."""
    complete_response = {
        "events": [],
        "model": "claude-3-7-sonnet-20250219",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 25,
            "output_tokens_details": {"reasoning_tokens": 15},
        },
        "id": "msg_1",
    }

    with tracer.start_as_current_span("test") as span:
        _set_streaming_token_usage(span, complete_response, 10, 25)

    attributes = _finished_span_attributes(span_exporter)
    assert attributes[REASONING_TOKENS] == 15
