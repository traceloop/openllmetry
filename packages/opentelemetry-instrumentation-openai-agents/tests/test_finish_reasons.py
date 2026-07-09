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
        """When trace_content=False, output messages are omitted but top-level
        finish_reasons must still reflect all output item types, including tool calls."""
        response = _make_response(
            output=[
                _text_message("secret"),
                _tool_call("get_weather", '{"city": "NYC"}', "call_abc"),
            ],
            finish_reason="stop",
        )

        _extract(span, response, trace_content=False)

        # No output messages (content suppressed)
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
        # finish_reasons must reflect both the text (stop) and the tool call (tool_call)
        assert _get_finish_reasons(span) == ("stop", "tool_call")

    def test_incomplete_response_preserves_incomplete_finish_reason(self, span):
        """Responses API status='incomplete' must preserve 'incomplete', not remap to 'length'.

        'incomplete' can be caused by a content filter, not just token limits — mapping
        to 'length' would misrepresent the reason and lose information.
        """
        response = _make_response(
            output=[_text_message("Partial...")],
            status="incomplete",
        )

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert msgs[0]["finish_reason"] == "incomplete"
        assert _get_finish_reasons(span) == ("incomplete",)

    def test_cancelled_response_preserves_cancelled_finish_reason(self, span):
        """Responses API status='cancelled' must preserve 'cancelled', not remap to 'error'.

        Cancellation is a distinct lifecycle event from an error; conflating the two
        prevents consumers from distinguishing user-initiated cancels from failures.
        """
        response = _make_response(
            output=[_text_message("Partial")],
            status="cancelled",
        )

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert msgs[0]["finish_reason"] == "cancelled"
        assert _get_finish_reasons(span) == ("cancelled",)

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


class TestFinishReasonsWithoutContent:
    """finish_reason granularity must be preserved when trace_content=False.

    gen_ai.response.finish_reasons is Recommended metadata, not opt-in content.
    The code must iterate output items for their inherent finish reasons even
    when message content is suppressed.
    """

    def test_tool_call_only_output_with_responses_api_completed_status(self, span):
        """Responses API status='completed' maps to 'stop' at the response level.
        But when the output contains only a function_call item, the top-level
        finish_reasons must show 'tool_call', not the response-level 'stop'."""
        response = _make_response(
            output=[_tool_call("get_weather", '{"city": "NYC"}', "call_abc")],
            status="completed",
        )

        _extract(span, response, trace_content=False)

        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
        assert _get_finish_reasons(span) == ("tool_call",)

    def test_tool_call_only_without_content_trace_enabled(self, span):
        """Same response with trace_content=True must also yield only 'tool_call'."""
        response = _make_response(
            output=[_tool_call("search", '{"q": "test"}', "call_1")],
            status="completed",
        )

        _extract(span, response, trace_content=True)

        msgs = _get_output_messages(span)
        assert msgs[0]["finish_reason"] == "tool_call"
        assert _get_finish_reasons(span) == ("tool_call",)

    def test_no_output_falls_back_to_response_level_finish_reason(self, span):
        """When the response has no output items, fall back to the response-level reason."""
        response = _make_response(output=[], finish_reason="stop")

        _extract(span, response, trace_content=False)

        assert _get_finish_reasons(span) == ("stop",)


class TestToolCallPartOptionalFields:
    """P2: Optional fields on tool_call parts must be omitted (not set to '' or null)
    when not present in the source data.

    OTel ToolCallRequestPart schema: id is optional (default null), arguments is optional.
    Emitting empty-string id or null arguments causes consumer correlation failures.
    """

    def test_function_call_no_call_id_omits_id_from_part(self, span):
        """Responses API function_call with no call_id must omit 'id' from the part."""
        output = SimpleNamespace(type="function_call", name="search", arguments='{"q": "test"}')
        # No call_id attribute — getattr fallback was "", which is wrong
        response = _make_response(output=[output], finish_reason="stop")

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert len(msgs) == 1
        part = msgs[0]["parts"][0]
        assert part["type"] == "tool_call"
        assert "id" not in part or part["id"], (
            f"id must be absent or non-empty when call_id not provided, got: {part}"
        )

    def test_function_call_none_arguments_omits_arguments_key(self, span):
        """Responses API function_call with no arguments must omit 'arguments' from the part."""
        output = SimpleNamespace(type="function_call", name="noop", call_id="c1")
        # No arguments attribute
        response = _make_response(output=[output], finish_reason="stop")

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert len(msgs) == 1
        part = msgs[0]["parts"][0]
        assert part["type"] == "tool_call"
        assert "arguments" not in part, (
            f"arguments must be omitted when None, got: {part}"
        )

    def test_function_call_with_call_id_and_arguments_still_included(self, span):
        """Sanity: when call_id and arguments are present, both must be emitted."""
        output = SimpleNamespace(
            type="function_call", name="get_weather",
            call_id="call_99", arguments='{"city": "NYC"}',
        )
        response = _make_response(output=[output], finish_reason="stop")

        _extract(span, response)

        msgs = _get_output_messages(span)
        part = msgs[0]["parts"][0]
        assert part.get("id") == "call_99"
        assert isinstance(part.get("arguments"), dict)
        assert part["arguments"]["city"] == "NYC"


class TestContentItemTypeClassification:
    """Content items inside output.content must be dispatched by their 'type' field first.

    The hasattr(.text) fallback must NOT shadow typed items like 'reasoning' or 'refusal'
    that happen to also carry a .text attribute.
    """

    def test_reasoning_item_with_text_attribute_not_misclassified_as_text(self, span):
        """A content item with type='reasoning' that also has a .text attribute
        must produce a 'reasoning' part, not a 'text' part."""
        from types import SimpleNamespace

        reasoning_item = SimpleNamespace(
            type="reasoning",
            text="This shadows the type if hasattr fires first",
            summary=[SimpleNamespace(text="actual chain-of-thought")],
        )
        output = SimpleNamespace(type="message", content=[reasoning_item], role="assistant")
        response = _make_response(output=[output], finish_reason="stop")

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert len(msgs) == 1
        part = msgs[0]["parts"][0]
        assert part["type"] == "reasoning", (
            f"type='reasoning' item with .text was misclassified as '{part['type']}'"
        )

    def test_refusal_item_with_text_attribute_not_misclassified_as_text(self, span):
        """A content item with type='refusal' that also has a .text attribute
        must produce a 'refusal' part, not a 'text' part."""
        from types import SimpleNamespace

        refusal_item = SimpleNamespace(
            type="refusal",
            text="I cannot do that",
            refusal="I cannot do that",
        )
        output = SimpleNamespace(type="message", content=[refusal_item], role="assistant")
        response = _make_response(output=[output], finish_reason="stop")

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert len(msgs) == 1
        part = msgs[0]["parts"][0]
        assert part["type"] == "refusal", (
            f"type='refusal' item with .text was misclassified as '{part['type']}'"
        )

    def test_output_text_item_still_produces_text_part(self, span):
        """Sanity check: type='output_text' must still produce a 'text' part."""
        from types import SimpleNamespace

        text_item = SimpleNamespace(type="output_text", text="Hello!")
        output = SimpleNamespace(type="message", content=[text_item], role="assistant")
        response = _make_response(output=[output], finish_reason="stop")

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert msgs[0]["parts"][0]["type"] == "text"
        assert msgs[0]["parts"][0]["content"] == "Hello!"

    def test_unknown_typed_item_without_text_still_handled(self, span):
        """An item with an unknown type and no .text must fall through to the generic handler."""
        from types import SimpleNamespace

        unknown_item = SimpleNamespace(type="image_file", file_id="file_abc")
        output = SimpleNamespace(type="message", content=[unknown_item], role="assistant")
        response = _make_response(output=[output], finish_reason="stop")

        _extract(span, response)

        msgs = _get_output_messages(span)
        assert len(msgs) == 1
        assert msgs[0]["parts"][0]["type"] == "image_file"
