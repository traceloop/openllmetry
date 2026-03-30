"""Unit tests for content block → OTel parts mapping functions.

These are pure unit tests (no VCR cassettes, no boto3) that validate
the mapping logic in span_utils.py against the OTel GenAI semconv.
"""

import json

from opentelemetry.instrumentation.bedrock.span_utils import (
    _anthropic_content_to_parts,
    _converse_content_to_parts,
    _map_finish_reason,
    _output_message,
    _text_part,
)
from opentelemetry.instrumentation.bedrock.streaming_wrapper import StreamingWrapper


# ─── _anthropic_content_to_parts ───────────────────────────────────────────


class TestAnthropicContentToParts:
    """Tests for _anthropic_content_to_parts mapping."""

    def test_text_block(self):
        blocks = [{"type": "text", "text": "Hello world"}]
        parts = _anthropic_content_to_parts(blocks)
        assert parts == [{"type": "text", "content": "Hello world"}]

    def test_string_content(self):
        blocks = ["plain string"]
        parts = _anthropic_content_to_parts(blocks)
        assert parts == [{"type": "text", "content": "plain string"}]

    def test_tool_use_block(self):
        blocks = [
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "get_weather",
                "input": {"city": "NYC"},
            }
        ]
        parts = _anthropic_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call"
        assert parts[0]["name"] == "get_weather"
        assert parts[0]["id"] == "call_123"
        # arguments must be a dict (object), not a JSON string
        assert isinstance(parts[0]["arguments"], dict)
        assert parts[0]["arguments"] == {"city": "NYC"}

    def test_tool_result_block(self):
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "call_123",
                "content": "72°F, sunny",
            }
        ]
        parts = _anthropic_content_to_parts(blocks)
        assert parts == [
            {"type": "tool_call_response", "id": "call_123", "response": "72°F, sunny"}
        ]

    def test_image_base64_block(self):
        blocks = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgo=",
                },
            }
        ]
        parts = _anthropic_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "blob"
        assert parts[0]["modality"] == "image"
        assert parts[0]["mime_type"] == "image/png"
        assert parts[0]["content"] == "iVBORw0KGgo="

    def test_image_url_block(self):
        blocks = [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/image.png",
                },
            }
        ]
        parts = _anthropic_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "uri"
        assert parts[0]["modality"] == "image"
        assert parts[0]["uri"] == "https://example.com/image.png"

    def test_thinking_block_maps_to_reasoning_part(self):
        """#15: Anthropic thinking blocks must map to ReasoningPart."""
        blocks = [
            {
                "type": "thinking",
                "thinking": "Let me analyze the user's question step by step...",
            }
        ]
        parts = _anthropic_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "reasoning"
        assert parts[0]["content"] == "Let me analyze the user's question step by step..."

    def test_thinking_block_empty_content(self):
        """#15: Thinking block with empty content."""
        blocks = [{"type": "thinking", "thinking": ""}]
        parts = _anthropic_content_to_parts(blocks)
        assert parts == [{"type": "reasoning", "content": ""}]

    def test_thinking_block_missing_thinking_key(self):
        """#15: Thinking block without 'thinking' key should not crash."""
        blocks = [{"type": "thinking"}]
        parts = _anthropic_content_to_parts(blocks)
        assert parts == [{"type": "reasoning", "content": ""}]

    def test_mixed_content_with_thinking(self):
        """#15: Mixed content including text and thinking blocks."""
        blocks = [
            {"type": "thinking", "thinking": "I should check the weather API..."},
            {"type": "text", "text": "The weather in NYC is 72°F."},
        ]
        parts = _anthropic_content_to_parts(blocks)
        assert len(parts) == 2
        assert parts[0] == {"type": "reasoning", "content": "I should check the weather API..."}
        assert parts[1] == {"type": "text", "content": "The weather in NYC is 72°F."}

    def test_tool_use_string_input_parsed_as_object(self):
        """#19: When streaming accumulates tool_use input as string, it should be parsed."""
        blocks = [
            {
                "type": "tool_use",
                "id": "call_456",
                "name": "search",
                "input": '{"query": "weather"}',
            }
        ]
        parts = _anthropic_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call"
        # arguments should be a parsed dict, not a raw JSON string
        assert isinstance(parts[0]["arguments"], dict)
        assert parts[0]["arguments"] == {"query": "weather"}

    def test_tool_use_invalid_string_input_kept_as_string(self):
        """#19: Invalid JSON string in tool_use input should be kept as-is."""
        blocks = [
            {
                "type": "tool_use",
                "id": "call_789",
                "name": "search",
                "input": "not valid json {",
            }
        ]
        parts = _anthropic_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["arguments"] == "not valid json {"


# ─── _converse_content_to_parts ────────────────────────────────────────────


class TestConverseContentToParts:
    """Tests for _converse_content_to_parts mapping."""

    def test_text_block(self):
        blocks = [{"text": "Hello world"}]
        parts = _converse_content_to_parts(blocks)
        assert parts == [{"type": "text", "content": "Hello world"}]

    def test_tool_use_block(self):
        blocks = [
            {
                "toolUse": {
                    "name": "get_weather",
                    "toolUseId": "call_123",
                    "input": {"city": "NYC"},
                }
            }
        ]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call"
        assert parts[0]["name"] == "get_weather"
        assert parts[0]["id"] == "call_123"
        assert isinstance(parts[0]["arguments"], dict)
        assert parts[0]["arguments"] == {"city": "NYC"}

    def test_tool_result_block(self):
        blocks = [
            {
                "toolResult": {
                    "toolUseId": "call_123",
                    "content": [{"text": "72°F, sunny"}],
                }
            }
        ]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call_response"
        assert parts[0]["id"] == "call_123"

    def test_image_block_bytes_source(self):
        """#14: Converse API image blocks must map to BlobPart with required 'content' field."""
        blocks = [
            {
                "image": {
                    "format": "png",
                    "source": {"bytes": b"fake_image_bytes"},
                }
            }
        ]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "blob"
        assert parts[0]["modality"] == "image"
        assert parts[0]["mime_type"] == "image/png"
        # 'content' is required per OTel BlobPart schema; empty string when raw bytes can't be base64'd
        assert "content" in parts[0]

    def test_image_block_jpeg_format(self):
        """#14: JPEG image format mapping."""
        blocks = [
            {
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": b"fake_jpeg_bytes"},
                }
            }
        ]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "blob"
        assert parts[0]["modality"] == "image"
        assert parts[0]["mime_type"] == "image/jpeg"

    def test_video_block(self):
        """#18: Converse API video blocks must map to BlobPart with modality=video."""
        blocks = [
            {
                "video": {
                    "format": "mp4",
                    "source": {"bytes": b"fake_video_bytes"},
                }
            }
        ]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "blob"
        assert parts[0]["modality"] == "video"
        # 'content' is required per OTel BlobPart schema
        assert "content" in parts[0]

    def test_document_block(self):
        """#18: Converse API document blocks must map to a part with document info."""
        blocks = [
            {
                "document": {
                    "format": "pdf",
                    "name": "report.pdf",
                    "source": {"bytes": b"fake_pdf_bytes"},
                }
            }
        ]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "blob"
        assert parts[0]["modality"] == "document"
        assert parts[0]["mime_type"] == "application/pdf"
        assert parts[0]["content"] == "report.pdf"

    def test_document_block_csv(self):
        """Document format 'csv' maps to 'text/csv'."""
        blocks = [{"document": {"format": "csv", "name": "data.csv", "source": {"bytes": b"x"}}}]
        parts = _converse_content_to_parts(blocks)
        assert parts[0]["mime_type"] == "text/csv"

    def test_document_block_unknown_format(self):
        """Unknown document format falls back to 'application/<fmt>'."""
        blocks = [{"document": {"format": "odt", "name": "f.odt", "source": {"bytes": b"x"}}}]
        parts = _converse_content_to_parts(blocks)
        assert parts[0]["mime_type"] == "application/odt"

    def test_guard_content_block(self):
        blocks = [{"guardContent": {"text": {"text": "blocked"}}}]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "text"

    def test_string_content(self):
        blocks = ["plain string"]
        parts = _converse_content_to_parts(blocks)
        assert parts == [{"type": "text", "content": "plain string"}]

    def test_mixed_content(self):
        blocks = [
            {"text": "Here's what I found:"},
            {
                "toolUse": {
                    "name": "search",
                    "toolUseId": "t1",
                    "input": {"q": "test"},
                }
            },
        ]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "tool_call"


# ─── _map_finish_reason ────────────────────────────────────────────────────


class TestMapFinishReason:
    """Verify finish reason mapping."""

    def test_none_returns_none(self):
        assert _map_finish_reason(None) is None

    def test_empty_string_returns_none(self):
        assert _map_finish_reason("") is None

    def test_end_turn_maps_to_stop(self):
        assert _map_finish_reason("end_turn") == "stop"

    def test_stop_sequence_maps_to_stop(self):
        assert _map_finish_reason("stop_sequence") == "stop"

    def test_tool_use_maps_to_tool_call(self):
        assert _map_finish_reason("tool_use") == "tool_call"

    def test_max_tokens_maps_to_length(self):
        assert _map_finish_reason("max_tokens") == "length"

    def test_guardrail_intervened_maps_to_content_filter(self):
        assert _map_finish_reason("guardrail_intervened") == "content_filter"

    def test_unknown_value_passes_through(self):
        assert _map_finish_reason("some_new_reason") == "some_new_reason"

    def test_stop_passes_through(self):
        """'stop' is already a valid OTel value and is not in the map."""
        assert _map_finish_reason("stop") == "stop"

    def test_length_passes_through(self):
        """'length' is already a valid OTel value and is not in the map."""
        assert _map_finish_reason("length") == "length"


# ─── _output_message ───────────────────────────────────────────────────────


class TestOutputMessage:
    """OutputMessage schema: required=[role, parts, finish_reason] per OTel spec."""

    def test_with_finish_reason(self):
        msg = _output_message("assistant", [_text_part("hi")], "stop")
        assert msg["finish_reason"] == "stop"
        assert msg["role"] == "assistant"

    def test_without_finish_reason_defaults_to_empty_string(self):
        """finish_reason is required per spec; defaults to empty string when unavailable."""
        msg = _output_message("assistant", [_text_part("hi")], None)
        assert "finish_reason" in msg
        assert msg["finish_reason"] == ""

    def test_without_finish_reason_arg_defaults_to_empty_string(self):
        """finish_reason is required per spec; defaults to empty string when omitted."""
        msg = _output_message("assistant", [_text_part("hi")])
        assert "finish_reason" in msg
        assert msg["finish_reason"] == ""


# ─── P1-1 regression: Nova output → single OutputMessage ─────────────────


class TestNovaOutputMessageCollapse:
    """P1-1: Amazon Nova 'output' path must produce ONE OutputMessage with
    multiple parts (via _converse_content_to_parts), not one per content block."""

    def test_multi_text_blocks_become_single_message(self):
        """Multiple text content blocks → one OutputMessage with multiple text parts."""
        from opentelemetry.instrumentation.bedrock.span_utils import (
            _converse_content_to_parts,
        )

        content_blocks = [
            {"text": "First paragraph."},
            {"text": "Second paragraph."},
        ]
        parts = _converse_content_to_parts(content_blocks)
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "content": "First paragraph."}
        assert parts[1] == {"type": "text", "content": "Second paragraph."}

        msg = _output_message("assistant", parts, "stop")
        assert len(msg["parts"]) == 2
        assert msg["finish_reason"] == "stop"

    def test_mixed_text_and_tool_use_blocks(self):
        """Nova output with text + toolUse → single OutputMessage, multiple parts."""
        content_blocks = [
            {"text": "Let me look that up."},
            {"toolUse": {"name": "search", "toolUseId": "t1", "input": {"q": "weather"}}},
        ]
        parts = _converse_content_to_parts(content_blocks)
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "tool_call"
        assert parts[1]["name"] == "search"

    def test_single_text_block(self):
        """Single text block → one OutputMessage with one part (no regression)."""
        content_blocks = [{"text": "Hello"}]
        parts = _converse_content_to_parts(content_blocks)
        assert len(parts) == 1
        msg = _output_message("assistant", parts, "stop")
        assert len(msg["parts"]) == 1
        assert msg["parts"][0]["content"] == "Hello"


# ─── P1-2 regression: StreamingWrapper thinking_delta accumulation ────────


def _make_chunk(payload):
    """Helper: wrap a dict as a streaming chunk event."""
    return {"chunk": {"bytes": json.dumps(payload).encode()}}


class _IterableList(list):
    """A list subclass that supports arbitrary attribute assignment,
    needed because StreamingWrapper (ObjectProxy) proxies setattr
    to the wrapped object."""

    def __iter__(self):
        return super().__iter__()


class TestStreamingWrapperThinkingDelta:
    """P1-2: StreamingWrapper must accumulate thinking_delta blocks so that
    Anthropic extended thinking content is captured in the invoke_model
    streaming path."""

    def _build_stream(self, events):
        """Build a StreamingWrapper from a list of raw chunk dicts."""
        results = {}

        def callback(body):
            results["body"] = body

        wrapper = StreamingWrapper(_IterableList(events), stream_done_callback=callback)
        for _ in wrapper:
            pass
        return results.get("body", {})

    def test_thinking_delta_accumulated(self):
        """thinking_delta chunks must be concatenated into the content block."""
        events = [
            _make_chunk({"type": "message_start", "message": {
                "content": [], "usage": {}, "stop_reason": None,
            }}),
            _make_chunk({"type": "content_block_start", "content_block": {
                "type": "thinking", "thinking": "",
            }}),
            _make_chunk({"type": "content_block_delta", "delta": {
                "type": "thinking_delta", "thinking": "Step 1: ",
            }}),
            _make_chunk({"type": "content_block_delta", "delta": {
                "type": "thinking_delta", "thinking": "analyze the question.",
            }}),
            _make_chunk({"type": "content_block_start", "content_block": {
                "type": "text", "text": "",
            }}),
            _make_chunk({"type": "content_block_delta", "delta": {
                "text": "The answer is 42.",
            }}),
            _make_chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}}),
            _make_chunk({"type": "message_stop", "amazon-bedrock-invocationMetrics": {}}),
        ]
        body = self._build_stream(events)
        assert len(body["content"]) == 2
        assert body["content"][0]["thinking"] == "Step 1: analyze the question."
        assert body["content"][1]["text"] == "The answer is 42."

    def test_thinking_delta_empty(self):
        """Empty thinking deltas should not crash — just append empty strings."""
        events = [
            _make_chunk({"type": "message_start", "message": {
                "content": [], "usage": {},
            }}),
            _make_chunk({"type": "content_block_start", "content_block": {
                "type": "thinking", "thinking": "",
            }}),
            _make_chunk({"type": "content_block_delta", "delta": {
                "type": "thinking_delta", "thinking": "",
            }}),
            _make_chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}}),
            _make_chunk({"type": "message_stop", "amazon-bedrock-invocationMetrics": {}}),
        ]
        body = self._build_stream(events)
        assert body["content"][0]["thinking"] == ""

    def test_text_and_input_json_still_work(self):
        """Ensure text and input_json_delta handling are not broken."""
        events = [
            _make_chunk({"type": "message_start", "message": {
                "content": [], "usage": {},
            }}),
            _make_chunk({"type": "content_block_start", "content_block": {
                "type": "tool_use", "id": "t1", "name": "search",
            }}),
            _make_chunk({"type": "content_block_delta", "delta": {
                "type": "input_json_delta", "partial_json": '{"q":',
            }}),
            _make_chunk({"type": "content_block_delta", "delta": {
                "type": "input_json_delta", "partial_json": '"test"}',
            }}),
            _make_chunk({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}}),
            _make_chunk({"type": "message_stop", "amazon-bedrock-invocationMetrics": {}}),
        ]
        body = self._build_stream(events)
        assert body["content"][0]["input"] == '{"q":"test"}'
