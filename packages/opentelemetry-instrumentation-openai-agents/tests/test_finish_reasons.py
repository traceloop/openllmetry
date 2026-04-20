"""
Unit tests for finish_reason attribution on multi-output Responses API spans.

Tests _extract_response_attributes directly with mock response objects
to verify per-message finish_reason mapping and top-level dedup.
"""

import json
import pytest
from types import SimpleNamespace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.fixture
def span():
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    otel_span = tracer.start_span("test-span")
    yield otel_span
    otel_span.end()


def _make_response(output, finish_reason="stop", status=None, model="gpt-4o"):
    kwargs = dict(
        output=output,
        model=model,
        id="resp-test",
        temperature=None,
        max_output_tokens=None,
        top_p=None,
        frequency_penalty=None,
        usage=None,
    )
    if status is not None:
        kwargs["status"] = status
    else:
        kwargs["finish_reason"] = finish_reason
    return SimpleNamespace(**kwargs)


def _text_message(*texts, role="assistant"):
    content = [
        SimpleNamespace(type="output_text", text=t) for t in texts
    ]
    return SimpleNamespace(type="message", content=content, role=role)


def _reasoning_and_text_message(reasoning_summary, text, role="assistant"):
    content = [
        SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=reasoning_summary)]),
        SimpleNamespace(type="output_text", text=text),
    ]
    return SimpleNamespace(type="message", content=content, role=role)


def _tool_call(name, arguments="{}", call_id="call_0"):
    return SimpleNamespace(
        type="function_call", name=name, arguments=arguments, call_id=call_id,
    )


def _extract(span, response, trace_content=True):
    from opentelemetry.instrumentation.openai_agents._hooks import (
        _extract_response_attributes,
    )
    return _extract_response_attributes(span, response, trace_content)


def _get_output_messages(span):
    raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
    return json.loads(raw) if raw else []


def _get_finish_reasons(span):
    return span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)


class TestMultiOutputFinishReasons:
    """Verify per-message and top-level finish_reasons when a Responses-API
    reply contains reasoning + text + tool_call outputs."""

    def test_reasoning_text_and_tool_call(self, span):
        response = _make_response(
            output=[
                _reasoning_and_text_message("let me think", "Here's what I found"),
                _tool_call("get_weather", '{"city": "NYC"}', "call_abc"),
            ],
            finish_reason="stop",
        )

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert len(msgs) == 2

        # Message 0: text+reasoning → mapped "stop"
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["finish_reason"] == "stop"
        part_types = [p["type"] for p in msgs[0]["parts"]]
        assert "reasoning" in part_types
        assert "text" in part_types

        # Message 1: tool call → mapped "tool_call" (singular, per OTel spec)
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["finish_reason"] == "tool_call"
        assert msgs[1]["parts"][0]["type"] == "tool_call"
        assert msgs[1]["parts"][0]["name"] == "get_weather"

        # Top-level: deduped, order-preserved
        assert _get_finish_reasons(span) == ("stop", "tool_call")

    def test_tool_calls_mapped_to_singular(self, span):
        """'tool_calls' (OpenAI) must map to 'tool_call' (OTel singular)."""
        response = _make_response(
            output=[_tool_call("search", '{"q": "test"}', "call_1")],
            finish_reason="stop",
        )

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert msgs[0]["finish_reason"] == "tool_call"
        assert _get_finish_reasons(span) == ("tool_call",)

    def test_status_completed_mapped_to_stop(self, span):
        """Responses API status='completed' must map to 'stop'."""
        response = _make_response(
            output=[_text_message("Done")],
            status="completed",
        )

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert msgs[0]["finish_reason"] == "stop"
        assert _get_finish_reasons(span) == ("stop",)

    def test_multiple_text_messages_dedup_finish_reason(self, span):
        """Two text outputs with the same finish_reason should dedup to one top-level entry."""
        response = _make_response(
            output=[
                _text_message("Part 1"),
                _text_message("Part 2"),
            ],
            finish_reason="stop",
        )

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert len(msgs) == 2
        assert msgs[0]["finish_reason"] == "stop"
        assert msgs[1]["finish_reason"] == "stop"
        # Deduped: only one "stop"
        assert _get_finish_reasons(span) == ("stop",)

    def test_text_and_tool_call_distinct_reasons(self, span):
        """Text ('stop') + tool call ('tool_call') → both in top-level tuple."""
        response = _make_response(
            output=[
                _text_message("Here you go"),
                _tool_call("lookup", '{"id": 1}', "call_2"),
            ],
            finish_reason="stop",
        )

        _extract(span, response)

        assert _get_finish_reasons(span) == ("stop", "tool_call")

    def test_trace_content_false_still_sets_finish_reasons(self, span):
        """When trace_content=False, output messages are omitted but
        top-level finish_reasons must still be set from mapped finish_reason."""
        response = _make_response(
            output=[
                _text_message("secret"),
                _tool_call("get_weather", '{"city": "NYC"}', "call_abc"),
            ],
            finish_reason="stop",
        )

        _extract(span, response, trace_content=False)

        # No output messages
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
        # But finish_reasons still present (from mapped response-level reason)
        assert _get_finish_reasons(span) == ("stop",)

    def test_incomplete_status_mapped_to_length(self, span):
        """Responses API status='incomplete' must map to 'length'."""
        response = _make_response(
            output=[_text_message("Partial...")],
            status="incomplete",
        )

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert msgs[0]["finish_reason"] == "length"
        assert _get_finish_reasons(span) == ("length",)

    def test_failed_status_mapped_to_error(self, span):
        """Responses API status='failed' must map to 'error'."""
        response = _make_response(
            output=[_text_message("Oops")],
            status="failed",
        )

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert msgs[0]["finish_reason"] == "error"
        assert _get_finish_reasons(span) == ("error",)
