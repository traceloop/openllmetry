"""
Unit tests for span_utils helpers.

Covers:
  _content_to_parts, _tool_calls_to_parts, and every set_* span-attribute function.
All tests use mock spans — no network calls, no cassettes.
"""

import json
from unittest.mock import MagicMock

import pytest

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes

from opentelemetry.instrumentation.groq.span_utils import (
    _content_to_parts,
    _tool_calls_to_parts,
    set_input_attributes,
    set_model_input_attributes,
    set_model_response_attributes,
    set_model_streaming_response_attributes,
    set_response_attributes,
    set_streaming_response_attributes,
)
from opentelemetry.instrumentation.groq.utils import TRACELOOP_TRACE_CONTENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span(recording: bool = True) -> MagicMock:
    s = MagicMock()
    s.is_recording.return_value = recording
    return s


def _attr(span: MagicMock, name: str):
    """Return the value passed to span.set_attribute(name, …)."""
    for call in span.set_attribute.call_args_list:
        if call[0][0] == name:
            return call[0][1]
    return None


# ---------------------------------------------------------------------------
# _content_to_parts
# ---------------------------------------------------------------------------


class TestContentToParts:
    def test_none_returns_empty(self):
        assert _content_to_parts(None) == []

    def test_empty_string_returns_empty(self):
        assert _content_to_parts("") == []

    def test_plain_string_returns_text_part(self):
        assert _content_to_parts("hello") == [{"type": "text", "content": "hello"}]

    def test_list_with_text_block(self):
        content = [{"type": "text", "text": "hello world"}]
        assert _content_to_parts(content) == [{"type": "text", "content": "hello world"}]

    def test_list_with_image_url_block(self):
        content = [{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}]
        assert _content_to_parts(content) == [
            {"type": "uri", "modality": "image", "uri": "https://example.com/img.jpg"}
        ]

    def test_list_with_mixed_content(self):
        content = [
            {"type": "text", "text": "Describe:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
        ]
        result = _content_to_parts(content)
        assert len(result) == 2
        assert result[0] == {"type": "text", "content": "Describe:"}
        assert result[1] == {"type": "uri", "modality": "image", "uri": "https://example.com/cat.jpg"}

    def test_list_with_non_dict_items_skipped(self):
        content = ["not a dict", {"type": "text", "text": "hello"}]
        assert _content_to_parts(content) == [{"type": "text", "content": "hello"}]

    def test_list_with_unknown_block_type_preserved_as_generic(self):
        content = [{"type": "audio", "data": "..."}]
        assert _content_to_parts(content) == [{"type": "audio", "data": "..."}]

    def test_list_with_data_url_image(self):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC123"}}]
        assert _content_to_parts(content) == [
            {"type": "blob", "modality": "image", "mime_type": "image/png", "content": "ABC123"}
        ]


# ---------------------------------------------------------------------------
# _tool_calls_to_parts
# ---------------------------------------------------------------------------


class TestToolCallsToParts:
    def test_none_returns_empty(self):
        assert _tool_calls_to_parts(None) == []

    def test_empty_list_returns_empty(self):
        assert _tool_calls_to_parts([]) == []

    def test_string_arguments_parsed_as_json(self):
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
            }
        ]
        result = _tool_calls_to_parts(tool_calls)
        assert result == [
            {
                "type": "tool_call",
                "id": "call_123",
                "name": "get_weather",
                "arguments": {"location": "SF"},
            }
        ]

    def test_dict_arguments_used_as_is(self):
        tool_calls = [
            {
                "id": "call_456",
                "type": "function",
                "function": {"name": "add", "arguments": {"a": 1, "b": 2}},
            }
        ]
        result = _tool_calls_to_parts(tool_calls)
        assert result[0]["arguments"] == {"a": 1, "b": 2}

    def test_invalid_json_string_returns_raw(self):
        tool_calls = [
            {
                "id": "call_789",
                "type": "function",
                "function": {"name": "foo", "arguments": "not valid json {{"},
            }
        ]
        result = _tool_calls_to_parts(tool_calls)
        assert result[0]["arguments"] == "not valid json {{"

    def test_no_arguments_omits_arguments_key(self):
        tool_calls = [
            {
                "id": "call_000",
                "type": "function",
                "function": {"name": "ping"},
            }
        ]
        result = _tool_calls_to_parts(tool_calls)
        assert "arguments" not in result[0]

    def test_non_dict_items_skipped(self):
        tool_calls = [
            "not a dict",
            {"id": "call_ok", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
        ]
        result = _tool_calls_to_parts(tool_calls)
        assert len(result) == 1
        assert result[0]["name"] == "foo"

    def test_pydantic_style_object_handled(self):
        """Groq SDK may return tool_calls as Pydantic objects, not dicts."""
        fn = MagicMock()
        fn.name = "get_weather"
        fn.arguments = '{"city": "Paris"}'

        tc = MagicMock()
        tc.id = "call_pydantic"
        tc.function = fn

        result = _tool_calls_to_parts([tc])
        assert len(result) == 1
        assert result[0]["id"] == "call_pydantic"
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"] == {"city": "Paris"}

    def test_multiple_tool_calls(self):
        tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": '{"x": 1}'}},
            {"id": "call_2", "type": "function", "function": {"name": "func2", "arguments": '{"y": 2}'}},
        ]
        result = _tool_calls_to_parts(tool_calls)
        assert len(result) == 2
        assert result[0]["name"] == "func1"
        assert result[1]["name"] == "func2"


# ---------------------------------------------------------------------------
# set_input_attributes
# ---------------------------------------------------------------------------


class TestSetInputAttributes:
    def test_non_recording_span_returns_early(self):
        span = _span(recording=False)
        set_input_attributes(span, {"messages": [{"role": "user", "content": "hello"}]})
        span.set_attribute.assert_not_called()

    def test_empty_messages_does_not_set_attribute(self):
        span = _span()
        set_input_attributes(span, {"messages": []})
        span.set_attribute.assert_not_called()

    def test_tool_role_creates_tool_call_response_part(self):
        span = _span()
        kwargs = {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "20°C in SF",
                }
            ]
        }
        set_input_attributes(span, kwargs)
        value = _attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert value is not None
        messages = json.loads(value)
        assert messages[0]["role"] == "tool"
        assert messages[0]["parts"] == [
            {
                "type": "tool_call_response",
                "id": "call_123",
                "response": "20°C in SF",
            }
        ]

    def test_message_with_tool_calls_adds_tool_call_parts(self):
        span = _span()
        kwargs = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_456",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                        }
                    ],
                }
            ]
        }
        set_input_attributes(span, kwargs)
        value = _attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert value is not None
        messages = json.loads(value)
        parts = messages[0]["parts"]
        assert any(p["type"] == "tool_call" and p["name"] == "get_weather" for p in parts)


# ---------------------------------------------------------------------------
# set_model_input_attributes
# ---------------------------------------------------------------------------


class TestSetModelInputAttributes:
    def test_non_recording_span_returns_early(self):
        span = _span(recording=False)
        set_model_input_attributes(span, {"model": "llama3-8b-8192"})
        span.set_attribute.assert_not_called()

    def test_with_tools_sets_tool_definitions(self):
        span = _span()
        tools = [{"type": "function", "function": {"name": "ping", "description": "Ping"}}]
        set_model_input_attributes(span, {"model": "llama3-8b-8192", "tools": tools})
        value = _attr(span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)
        assert value is not None
        assert json.loads(value) == tools

    def test_unserializable_tools_silently_ignored(self):
        span = _span()
        # object() is not JSON-serializable — should not raise
        set_model_input_attributes(span, {"model": "llama3-8b-8192", "tools": [object()]})
        # GEN_AI_TOOL_DEFINITIONS must NOT be set (exception swallowed)
        assert _attr(span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS) is None

    def test_tools_not_set_when_send_prompts_disabled(self):
        """Tool definitions are Opt-In — must NOT be recorded when content tracing is off."""
        span = _span()
        tools = [{"type": "function", "function": {"name": "ping"}}]
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv(TRACELOOP_TRACE_CONTENT, "False")
            set_model_input_attributes(span, {"model": "llama3-8b-8192", "tools": tools})
        assert _attr(span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS) is None


# ---------------------------------------------------------------------------
# set_streaming_response_attributes
# ---------------------------------------------------------------------------


class TestSetStreamingResponseAttributes:
    def test_non_recording_span_returns_early(self):
        span = _span(recording=False)
        set_streaming_response_attributes(span, "some content")
        span.set_attribute.assert_not_called()

    def test_empty_content_produces_empty_parts(self):
        span = _span()
        set_streaming_response_attributes(span, "", finish_reason="stop")
        value = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(value)
        assert messages[0]["parts"] == []
        assert messages[0]["finish_reason"] == "stop"

    def test_with_content_produces_text_part(self):
        span = _span()
        set_streaming_response_attributes(span, "Hello!", finish_reason="stop")
        value = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(value)
        assert messages[0]["parts"] == [{"type": "text", "content": "Hello!"}]
        assert messages[0]["role"] == "assistant"

    def test_with_tool_calls_adds_tool_call_parts(self):
        span = _span()
        tool_calls = [{"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
        set_streaming_response_attributes(span, "", finish_reason="tool_calls", tool_calls=tool_calls)
        value = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(value)
        parts = messages[0]["parts"]
        assert any(p["type"] == "tool_call" and p["name"] == "get_weather" for p in parts)
        assert messages[0]["finish_reason"] == "tool_call"


# ---------------------------------------------------------------------------
# set_model_streaming_response_attributes
# ---------------------------------------------------------------------------


class TestSetModelStreamingResponseAttributes:
    def test_non_recording_span_returns_early(self):
        span = _span(recording=False)
        usage = MagicMock()
        set_model_streaming_response_attributes(span, usage)
        span.set_attribute.assert_not_called()

    def test_with_usage_sets_token_counts(self):
        span = _span()
        usage = MagicMock()
        usage.completion_tokens = 25
        usage.prompt_tokens = 55
        usage.total_tokens = 80
        set_model_streaming_response_attributes(span, usage)
        assert _attr(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 25
        assert _attr(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 55

    def test_none_usage_skips_token_counts(self):
        span = _span()
        set_model_streaming_response_attributes(span, None)
        set_keys = [c[0][0] for c in span.set_attribute.call_args_list]
        assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in set_keys

    def test_with_finish_reason_sets_mapped_reason(self):
        span = _span()
        set_model_streaming_response_attributes(span, None, finish_reasons=["stop"])
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_none_finish_reason_skips_finish_reasons(self):
        span = _span()
        set_model_streaming_response_attributes(span, None, finish_reasons=None)
        set_keys = [c[0][0] for c in span.set_attribute.call_args_list]
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in set_keys

    def test_unknown_finish_reason_is_preserved(self):
        span = _span()
        set_model_streaming_response_attributes(span, None, finish_reasons=["unknown_reason_xyz"])
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["unknown_reason_xyz"]


# ---------------------------------------------------------------------------
# set_model_response_attributes
# ---------------------------------------------------------------------------


class TestSetModelResponseAttributes:
    """
    Uses plain dicts as response so model_as_dict() returns them unchanged.
    _collect_finish_reasons_from_response() uses getattr(response, "choices", None)
    which returns None for dicts → reasons = [] (covers the 'if reasons:' False branch).
    """

    def _response(self, prompt_tokens=18, completion_tokens=5):
        return {
            "id": "chatcmpl-test",
            "model": "llama3-8b-8192",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def test_non_recording_span_returns_early(self):
        span = _span(recording=False)
        histogram = MagicMock()
        set_model_response_attributes(span, self._response(), histogram)
        span.set_attribute.assert_not_called()
        histogram.record.assert_not_called()

    def test_response_without_usage_skips_histogram(self):
        span = _span()
        histogram = MagicMock()
        response = {"id": "chatcmpl-test", "model": "llama3-8b-8192"}
        set_model_response_attributes(span, response, histogram)
        histogram.record.assert_not_called()

    def test_none_histogram_does_not_raise(self):
        span = _span()
        set_model_response_attributes(span, self._response(), None)
        # Tokens are still set on span, just not recorded in histogram
        assert _attr(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18

    def test_histogram_records_input_and_output_tokens(self):
        span = _span()
        histogram = MagicMock()
        set_model_response_attributes(span, self._response(prompt_tokens=10, completion_tokens=5), histogram)
        assert histogram.record.call_count == 2
        first = histogram.record.call_args_list[0]
        assert first[0][0] == 10
        assert first[1]["attributes"]["gen_ai.token.type"] == "input"
        second = histogram.record.call_args_list[1]
        assert second[0][0] == 5
        assert second[1]["attributes"]["gen_ai.token.type"] == "output"


# ---------------------------------------------------------------------------
# set_response_attributes
# ---------------------------------------------------------------------------


class TestSetResponseAttributes:
    def test_non_recording_span_returns_early(self):
        span = _span(recording=False)
        response = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
        set_response_attributes(span, response)
        span.set_attribute.assert_not_called()

    def test_empty_choices_does_not_set_attribute(self):
        span = _span()
        set_response_attributes(span, {"choices": []})
        span.set_attribute.assert_not_called()

    def test_with_tool_calls_in_response(self):
        span = _span()
        response = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                            }
                        ],
                    },
                }
            ]
        }
        set_response_attributes(span, response)
        value = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(value)
        assert messages[0]["finish_reason"] == "tool_call"
        parts = messages[0]["parts"]
        assert any(p["type"] == "tool_call" and p["name"] == "get_weather" for p in parts)

    def test_with_content_filter(self):
        span = _span()
        response = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "content_filter",
                    "message": {"role": "assistant", "content": "..."},
                }
            ]
        }
        set_response_attributes(span, response)
        value = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(value)
        assert messages[0]["finish_reason"] == "content_filter"
        assert messages[0]["parts"] == [{"type": "text", "content": "..."}]

    def test_with_legacy_function_call(self):
        span = _span()
        response = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    },
                }
            ]
        }
        set_response_attributes(span, response)
        value = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(value)
        parts = messages[0]["parts"]
        assert any(p["type"] == "tool_call" and p["name"] == "get_weather" for p in parts)

    def test_legacy_function_call_with_invalid_json_arguments(self):
        span = _span()
        response = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"name": "foo", "arguments": "not valid json {{"},
                    },
                }
            ]
        }
        set_response_attributes(span, response)
        value = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(value)
        part = next(p for p in messages[0]["parts"] if p["type"] == "tool_call")
        assert part["arguments"] == "not valid json {{"

    def test_legacy_function_call_with_dict_arguments(self):
        span = _span()
        response = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"name": "bar", "arguments": {"x": 42}},
                    },
                }
            ]
        }
        set_response_attributes(span, response)
        value = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(value)
        part = next(p for p in messages[0]["parts"] if p["type"] == "tool_call")
        assert part["arguments"] == {"x": 42}

    def test_no_prompts_does_not_set_attribute(self):
        span = _span()
        response = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv(TRACELOOP_TRACE_CONTENT, "False")
            set_response_attributes(span, response)
        span.set_attribute.assert_not_called()
