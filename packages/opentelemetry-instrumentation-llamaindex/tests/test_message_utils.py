"""Unit tests for _message_utils — pure message-building functions."""

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.llamaindex._message_utils import (
    _content_to_parts,
    _extract_tool_calls,
    _parse_arguments,
    build_completion_output_message,
    build_input_messages,
    build_output_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role, content, **additional_kwargs):
    """Create a fake ChatMessage-like object."""
    m = SimpleNamespace(role=SimpleNamespace(value=role), content=content, additional_kwargs=additional_kwargs)
    return m


# ===========================================================================
# _parse_arguments
# ===========================================================================

class TestParseArguments:
    def test_none(self):
        assert _parse_arguments(None) is None

    def test_dict_passthrough(self):
        d = {"key": "val"}
        assert _parse_arguments(d) is d

    def test_json_string(self):
        assert _parse_arguments('{"a": 1}') == {"a": 1}

    def test_invalid_json_string_passthrough(self):
        assert _parse_arguments("not json") == "not json"

    def test_other_type_passthrough(self):
        assert _parse_arguments(42) == 42


# ===========================================================================
# _content_to_parts
# ===========================================================================

class TestContentToParts:
    def test_none_returns_empty(self):
        assert _content_to_parts(None) == []

    def test_empty_string_returns_empty(self):
        assert _content_to_parts("") == []

    def test_string_returns_text_part(self):
        assert _content_to_parts("hello") == [{"type": "text", "content": "hello"}]

    def test_list_of_strings(self):
        parts = _content_to_parts(["a", "b"])
        assert parts == [
            {"type": "text", "content": "a"},
            {"type": "text", "content": "b"},
        ]

    def test_list_with_text_block(self):
        parts = _content_to_parts([{"type": "text", "text": "hi"}])
        assert parts == [{"type": "text", "content": "hi"}]

    def test_image_url_block(self):
        block = {"type": "image_url", "image_url": {"url": "https://img.png"}}
        parts = _content_to_parts([block])
        assert parts == [{"type": "uri", "modality": "image", "uri": "https://img.png"}]

    def test_image_base64_block(self):
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
        }
        parts = _content_to_parts([block])
        assert parts == [
            {"type": "blob", "modality": "image", "mime_type": "image/png", "content": "abc123"}
        ]

    def test_blob_part_uses_content_key_not_data(self):
        """OTel BlobPart schema requires 'content' for base64 data, not 'data'."""
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "base64data"},
        }
        parts = _content_to_parts([block])
        assert "content" in parts[0], "BlobPart must use 'content' key per OTel spec"
        assert "data" not in parts[0], "BlobPart must NOT use 'data' key"
        assert parts[0]["content"] == "base64data"

    def test_image_url_source_block(self):
        block = {"type": "image", "source": {"type": "url", "url": "https://img.png"}}
        parts = _content_to_parts([block])
        assert parts == [{"type": "uri", "modality": "image", "uri": "https://img.png"}]

    def test_mixed_text_and_image(self):
        blocks = [
            {"type": "text", "text": "Look at this:"},
            {"type": "image_url", "image_url": {"url": "https://img.png"}},
        ]
        parts = _content_to_parts(blocks)
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "content": "Look at this:"}
        assert parts[1]["type"] == "uri"

    def test_thinking_block_maps_to_reasoning_part(self):
        """Anthropic-style thinking blocks must emit ReasoningPart, not TextPart."""
        block = {"type": "thinking", "thinking": "Let me think step by step..."}
        parts = _content_to_parts([block])
        assert parts == [{"type": "reasoning", "content": "Let me think step by step..."}]

    def test_reasoning_block_maps_to_reasoning_part(self):
        """Generic reasoning blocks must emit ReasoningPart."""
        block = {"type": "reasoning", "content": "Step 1: analyze the problem"}
        parts = _content_to_parts([block])
        assert parts == [{"type": "reasoning", "content": "Step 1: analyze the problem"}]

    def test_thinking_block_with_content_key(self):
        """Thinking block using 'content' key instead of 'thinking' key."""
        block = {"type": "thinking", "content": "Deep thought..."}
        parts = _content_to_parts([block])
        assert parts == [{"type": "reasoning", "content": "Deep thought..."}]

    def test_thinking_block_with_text_key(self):
        """Thinking block using 'text' key as fallback."""
        block = {"type": "thinking", "text": "Reasoning text"}
        parts = _content_to_parts([block])
        assert parts == [{"type": "reasoning", "content": "Reasoning text"}]

    def test_fallback_dict_with_text_key(self):
        parts = _content_to_parts([{"type": "custom", "text": "fallback"}])
        assert parts == [{"type": "text", "content": "fallback"}]

    def test_fallback_dict_with_content_key(self):
        parts = _content_to_parts([{"type": "custom", "content": 42}])
        assert parts == [{"type": "text", "content": "42"}]

    def test_non_str_non_list_stringified(self):
        parts = _content_to_parts(12345)
        assert parts == [{"type": "text", "content": "12345"}]


# ===========================================================================
# _extract_tool_calls
# ===========================================================================

class TestExtractToolCalls:
    def test_no_tool_calls(self):
        msg = _msg("assistant", "hi")
        assert _extract_tool_calls(msg) == []

    def test_single_tool_call(self):
        tc = {"id": "tc1", "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}}
        msg = _msg("assistant", "hi", tool_calls=[tc])
        parts = _extract_tool_calls(msg)
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call"
        assert parts[0]["id"] == "tc1"
        assert parts[0]["name"] == "get_weather"
        assert parts[0]["arguments"] == {"city": "NYC"}

    def test_skips_non_dict_tool_calls(self):
        msg = _msg("assistant", "hi", tool_calls=["not_a_dict"])
        assert _extract_tool_calls(msg) == []


# ===========================================================================
# build_input_messages
# ===========================================================================

class TestBuildInputMessages:
    def test_single_user_message(self):
        msgs = [_msg("user", "Hello")]
        result = build_input_messages(msgs)
        assert result == [{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]

    def test_multiple_messages_with_roles(self):
        msgs = [
            _msg("system", "You are helpful"),
            _msg("user", "Hi"),
            _msg("assistant", "Hello!"),
        ]
        result = build_input_messages(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_system_message_inline(self):
        msgs = [_msg("system", "Be concise")]
        result = build_input_messages(msgs)
        assert result[0]["parts"][0]["content"] == "Be concise"

    def test_message_with_none_content(self):
        msgs = [_msg("assistant", None)]
        result = build_input_messages(msgs)
        assert result == [{"role": "assistant", "parts": []}]

    def test_message_with_empty_content(self):
        msgs = [_msg("user", "")]
        result = build_input_messages(msgs)
        assert result == [{"role": "user", "parts": []}]

    def test_message_order_preserved(self):
        msgs = [_msg("user", f"msg{i}") for i in range(5)]
        result = build_input_messages(msgs)
        for i, m in enumerate(result):
            assert m["parts"][0]["content"] == f"msg{i}"

    def test_assistant_message_with_tool_calls(self):
        tc = {"id": "tc1", "function": {"name": "search", "arguments": '{"q": "test"}'}}
        msg = _msg("assistant", "Let me search", tool_calls=[tc])
        result = build_input_messages([msg])
        parts = result[0]["parts"]
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "tool_call"

    def test_tool_role_message(self):
        msg = _msg("tool", "result data", tool_call_id="tc1")
        result = build_input_messages([msg])
        parts = result[0]["parts"]
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call_response"
        assert parts[0]["id"] == "tc1"
        assert parts[0]["response"] == "result data"

    def test_tool_role_without_call_id_keeps_text(self):
        msg = _msg("tool", "result data")
        result = build_input_messages([msg])
        assert result[0]["parts"][0]["type"] == "text"

    def test_empty_messages_list(self):
        assert build_input_messages([]) == []

    def test_none_messages(self):
        assert build_input_messages(None) == []

    def test_multimodal_content_list(self):
        msg = _msg("user", [{"type": "text", "text": "Describe this"}, {"type": "image_url", "image_url": {"url": "https://img.png"}}])
        result = build_input_messages([msg])
        parts = result[0]["parts"]
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "uri"


# ===========================================================================
# build_output_message
# ===========================================================================

class TestBuildOutputMessage:
    def test_single_assistant_response(self):
        resp = _msg("assistant", "The answer is 42.")
        result = build_output_message(resp)
        assert result["role"] == "assistant"
        assert result["parts"] == [{"type": "text", "content": "The answer is 42."}]
        assert result["finish_reason"] == ""

    def test_response_with_none_content(self):
        resp = _msg("assistant", None)
        result = build_output_message(resp)
        assert result["parts"] == []

    def test_response_with_tool_call_parts(self):
        tc = {"id": "tc1", "function": {"name": "calc", "arguments": '{"x": 1}'}}
        resp = _msg("assistant", "Calling tool", tool_calls=[tc])
        result = build_output_message(resp, finish_reason="tool_calls")
        assert any(p["type"] == "tool_call" for p in result["parts"])


# ===========================================================================
# build_completion_output_message
# ===========================================================================

class TestBuildCompletionOutputMessage:
    def test_basic(self):
        result = build_completion_output_message("Hello world")
        assert result == {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Hello world"}],
            "finish_reason": "",
        }

    def test_empty_text(self):
        result = build_completion_output_message("")
        assert result["parts"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
