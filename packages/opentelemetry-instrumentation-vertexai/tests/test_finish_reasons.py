import json
import pytest
from unittest.mock import Mock, patch
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.instrumentation.vertexai.span_utils import (
    _map_vertex_finish_reason,
    accumulate_vertex_stream_finish_reasons,
    _output_messages_from_vertex_response,
    set_response_attributes,
    _parts_from_vertex_part_sync,
    _parts_from_vertex_part_async,
)


def _make_enum(name):
    """Create a simple mock object that mimics a VertexAI enum with a .name attribute."""
    obj = Mock()
    obj.name = name
    return obj


def _make_span():
    """Create a mock span that captures set_attribute calls."""
    span = Mock()
    span.is_recording.return_value = True
    span.context.trace_id = "test_trace_id"
    span.context.span_id = "test_span_id"
    attrs = {}

    def capture(key, value):
        attrs[key] = value

    span.set_attribute = capture
    span._attrs = attrs
    return span


# ---------------------------------------------------------------------------
# 1. _map_vertex_finish_reason
# ---------------------------------------------------------------------------
class TestMapVertexFinishReason:
    def test_none_returns_unknown(self):
        assert _map_vertex_finish_reason(None) == ""

    def test_stop_enum(self):
        assert _map_vertex_finish_reason(_make_enum("STOP")) == "stop"

    def test_max_tokens_enum(self):
        assert _map_vertex_finish_reason(_make_enum("MAX_TOKENS")) == "length"

    def test_safety_enum(self):
        assert _map_vertex_finish_reason(_make_enum("SAFETY")) == "content_filter"

    def test_recitation_enum(self):
        assert _map_vertex_finish_reason(_make_enum("RECITATION")) == "content_filter"

    def test_blocklist_enum(self):
        assert _map_vertex_finish_reason(_make_enum("BLOCKLIST")) == "content_filter"

    def test_prohibited_content_enum(self):
        assert _map_vertex_finish_reason(_make_enum("PROHIBITED_CONTENT")) == "content_filter"

    def test_spii_enum(self):
        assert _map_vertex_finish_reason(_make_enum("SPII")) == "content_filter"

    def test_image_safety_enum(self):
        assert _map_vertex_finish_reason(_make_enum("IMAGE_SAFETY")) == "content_filter"

    def test_image_prohibited_content_enum(self):
        assert _map_vertex_finish_reason(_make_enum("IMAGE_PROHIBITED_CONTENT")) == "content_filter"

    def test_image_recitation_enum(self):
        assert _map_vertex_finish_reason(_make_enum("IMAGE_RECITATION")) == "content_filter"

    def test_language_enum(self):
        assert _map_vertex_finish_reason(_make_enum("LANGUAGE")) == "content_filter"

    def test_finish_reason_unspecified_enum(self):
        assert _map_vertex_finish_reason(_make_enum("FINISH_REASON_UNSPECIFIED")) == ""

    def test_malformed_function_call_enum(self):
        assert _map_vertex_finish_reason(_make_enum("MALFORMED_FUNCTION_CALL")) == "error"

    def test_other_enum(self):
        assert _map_vertex_finish_reason(_make_enum("OTHER")) == "error"

    def test_unexpected_tool_call_enum(self):
        assert _map_vertex_finish_reason(_make_enum("UNEXPECTED_TOOL_CALL")) == "error"

    def test_no_image_enum(self):
        assert _map_vertex_finish_reason(_make_enum("NO_IMAGE")) == "error"

    def test_image_other_enum(self):
        assert _map_vertex_finish_reason(_make_enum("IMAGE_OTHER")) == "error"

    def test_unmapped_enum_returns_unknown(self):
        assert _map_vertex_finish_reason(_make_enum("TOTALLY_NEW_REASON")) == ""


# ---------------------------------------------------------------------------
# 2. accumulate_vertex_stream_finish_reasons
# ---------------------------------------------------------------------------
class TestAccumulateVertexStreamFinishReasons:
    def test_unknown_is_not_added(self):
        """Candidates with None finish_reason map to 'unknown' and are skipped."""
        ordered = []
        seen = set()
        chunk = Mock()
        cand = Mock()
        cand.finish_reason = None  # maps to ""
        chunk.candidates = [cand]

        accumulate_vertex_stream_finish_reasons(ordered, seen, chunk)

        assert ordered == []
        assert seen == set()

    def test_real_values_are_added(self):
        ordered = []
        seen = set()

        chunk1 = Mock()
        cand1 = Mock()
        cand1.finish_reason = _make_enum("STOP")
        chunk1.candidates = [cand1]

        accumulate_vertex_stream_finish_reasons(ordered, seen, chunk1)
        assert ordered == ["stop"]
        assert "stop" in seen

    def test_duplicates_are_skipped(self):
        ordered = []
        seen = set()

        for _ in range(3):
            chunk = Mock()
            cand = Mock()
            cand.finish_reason = _make_enum("STOP")
            chunk.candidates = [cand]
            accumulate_vertex_stream_finish_reasons(ordered, seen, chunk)

        assert ordered == ["stop"]

    def test_order_is_preserved(self):
        ordered = []
        seen = set()

        for name in ["STOP", "MAX_TOKENS", "SAFETY"]:
            chunk = Mock()
            cand = Mock()
            cand.finish_reason = _make_enum(name)
            chunk.candidates = [cand]
            accumulate_vertex_stream_finish_reasons(ordered, seen, chunk)

        assert ordered == ["stop", "length", "content_filter"]

    def test_chunk_with_no_candidates(self):
        ordered = []
        seen = set()
        chunk = Mock()
        chunk.candidates = None

        accumulate_vertex_stream_finish_reasons(ordered, seen, chunk)
        assert ordered == []


# ---------------------------------------------------------------------------
# 3. _output_messages_from_vertex_response
# ---------------------------------------------------------------------------
class TestOutputMessagesFromVertexResponse:
    def _make_candidate(self, finish_reason=None, text="Hello"):
        """Build a mock candidate with optional finish_reason and text."""
        part = Mock()
        part.text = text
        # No image, function_call, function_response, or thought attributes
        part.inline_data = None
        part.mime_type = None
        part.function_call = None
        part.function_response = None
        part.thought = None
        content = Mock()
        content.parts = [part]
        content.role = "model"
        cand = Mock()
        cand.content = content
        cand.finish_reason = finish_reason
        return cand

    def test_every_message_has_finish_reason_key(self):
        span = _make_span()
        response = Mock()
        response.candidates = [
            self._make_candidate(finish_reason=_make_enum("STOP")),
            self._make_candidate(finish_reason=_make_enum("MAX_TOKENS")),
        ]

        messages = _output_messages_from_vertex_response(span, response)

        for msg in messages:
            assert "finish_reason" in msg

    def test_finish_reason_is_always_string(self):
        span = _make_span()
        response = Mock()
        response.candidates = [
            self._make_candidate(finish_reason=_make_enum("STOP")),
            self._make_candidate(finish_reason=None),
        ]

        messages = _output_messages_from_vertex_response(span, response)

        for msg in messages:
            assert isinstance(msg["finish_reason"], str)
            assert msg["finish_reason"] is not None

    def test_no_finish_reason_returns_unknown(self):
        span = _make_span()
        response = Mock()
        response.candidates = [self._make_candidate(finish_reason=None)]

        messages = _output_messages_from_vertex_response(span, response)

        assert len(messages) == 1
        assert messages[0]["finish_reason"] == ""

    def test_stop_finish_reason(self):
        span = _make_span()
        response = Mock()
        response.candidates = [self._make_candidate(finish_reason=_make_enum("STOP"))]

        messages = _output_messages_from_vertex_response(span, response)

        assert len(messages) == 1
        assert messages[0]["finish_reason"] == "stop"

    def test_text_fallback_has_unknown_finish_reason(self):
        """When there are no candidates but response.text exists, finish_reason is 'unknown'."""
        span = _make_span()
        response = Mock()
        response.candidates = []
        response.text = "Fallback text"

        messages = _output_messages_from_vertex_response(span, response)

        assert len(messages) == 1
        assert messages[0]["finish_reason"] == ""
        assert messages[0]["parts"][0]["content"] == "Fallback text"


# ---------------------------------------------------------------------------
# 4. set_response_attributes
# ---------------------------------------------------------------------------
class TestSetResponseAttributes:
    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_full_response_object_with_candidates(self, mock_should_send):
        mock_should_send.return_value = True
        span = _make_span()

        part = Mock()
        part.text = "Generated text"
        part.inline_data = None
        part.mime_type = None
        part.function_call = None
        part.function_response = None
        part.thought = None
        content = Mock()
        content.parts = [part]
        content.role = "model"
        cand = Mock()
        cand.content = content
        cand.finish_reason = _make_enum("STOP")

        response = Mock()
        response.candidates = [cand]

        set_response_attributes(span, "gemini-pro", response)

        key = GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
        assert key in span._attrs
        messages = json.loads(span._attrs[key])
        assert len(messages) == 1
        assert messages[0]["finish_reason"] == "stop"
        assert messages[0]["role"] == "assistant"

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_plain_text_with_finish_reason_otel(self, mock_should_send):
        mock_should_send.return_value = True
        span = _make_span()

        set_response_attributes(span, "gemini-pro", "Hello world", finish_reason_otel="stop")

        key = GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
        assert key in span._attrs
        messages = json.loads(span._attrs[key])
        assert len(messages) == 1
        assert messages[0]["finish_reason"] == "stop"
        assert messages[0]["parts"][0]["content"] == "Hello world"

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_plain_text_without_finish_reason_otel(self, mock_should_send):
        mock_should_send.return_value = True
        span = _make_span()

        set_response_attributes(span, "gemini-pro", "Hello world")

        key = GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
        assert key in span._attrs
        messages = json.loads(span._attrs[key])
        assert len(messages) == 1
        assert messages[0]["finish_reason"] == ""

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_empty_text_no_finish_reason_no_output(self, mock_should_send):
        """Empty text + no finish_reason_otel => no output message emitted."""
        mock_should_send.return_value = True
        span = _make_span()

        set_response_attributes(span, "gemini-pro", "")

        key = GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
        assert key not in span._attrs

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_empty_text_with_finish_reason_emits_message(self, mock_should_send):
        """Empty text but with a real finish_reason_otel still emits a message."""
        mock_should_send.return_value = True
        span = _make_span()

        set_response_attributes(span, "gemini-pro", "", finish_reason_otel="stop")

        key = GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
        assert key in span._attrs
        messages = json.loads(span._attrs[key])
        assert len(messages) == 1
        assert messages[0]["finish_reason"] == "stop"
        assert messages[0]["parts"] == []

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_no_output_when_prompts_disabled(self, mock_should_send):
        mock_should_send.return_value = False
        span = _make_span()

        set_response_attributes(span, "gemini-pro", "Hello world", finish_reason_otel="stop")

        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span._attrs


# ---------------------------------------------------------------------------
# 5. _parts_from_vertex_part_sync — reasoning / thinking blocks
# ---------------------------------------------------------------------------
class TestPartsFromVertexPartSyncReasoning:
    def test_thought_only_produces_reasoning(self):
        """Part with thought attribute produces a reasoning-type output."""
        part = Mock()
        part.text = None
        part.inline_data = None
        part.mime_type = None
        part.function_call = None
        part.function_response = None
        part.thought = "Let me think about this..."

        span = _make_span()
        result = _parts_from_vertex_part_sync(part, span, 0)

        assert any(p["type"] == "reasoning" for p in result)
        reasoning = [p for p in result if p["type"] == "reasoning"][0]
        assert reasoning["content"] == "Let me think about this..."

    def test_text_and_thought_produces_both(self):
        """Part with both text and thought produces text and reasoning parts."""
        part = Mock()
        part.text = "The answer is 42."
        part.inline_data = None
        part.mime_type = None
        part.function_call = None
        part.function_response = None
        part.thought = "I need to calculate..."

        span = _make_span()
        result = _parts_from_vertex_part_sync(part, span, 0)

        types = [p["type"] for p in result]
        assert "text" in types
        assert "reasoning" in types
        text_part = [p for p in result if p["type"] == "text"][0]
        reasoning_part = [p for p in result if p["type"] == "reasoning"][0]
        assert text_part["content"] == "The answer is 42."
        assert reasoning_part["content"] == "I need to calculate..."

    def test_no_recognized_attributes_falls_through_to_str(self):
        """Part with no recognized attributes produces str(part) as text fallback."""
        part = Mock()
        part.text = None
        part.inline_data = None
        part.mime_type = None
        part.function_call = None
        part.function_response = None
        part.thought = None
        part.__str__ = lambda self: "fallback-string-repr"

        span = _make_span()
        result = _parts_from_vertex_part_sync(part, span, 0)

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["content"] == "fallback-string-repr"


# ---------------------------------------------------------------------------
# 6. _parts_from_vertex_part_async — same reasoning tests but async
# ---------------------------------------------------------------------------
class TestPartsFromVertexPartAsyncReasoning:
    @pytest.mark.asyncio
    async def test_thought_only_produces_reasoning(self):
        part = Mock()
        part.text = None
        part.inline_data = None
        part.mime_type = None
        part.function_call = None
        part.function_response = None
        part.thought = "Let me think about this..."

        span = _make_span()
        result = await _parts_from_vertex_part_async(part, span, 0)

        assert any(p["type"] == "reasoning" for p in result)
        reasoning = [p for p in result if p["type"] == "reasoning"][0]
        assert reasoning["content"] == "Let me think about this..."

    @pytest.mark.asyncio
    async def test_text_and_thought_produces_both(self):
        part = Mock()
        part.text = "The answer is 42."
        part.inline_data = None
        part.mime_type = None
        part.function_call = None
        part.function_response = None
        part.thought = "I need to calculate..."

        span = _make_span()
        result = await _parts_from_vertex_part_async(part, span, 0)

        types = [p["type"] for p in result]
        assert "text" in types
        assert "reasoning" in types
        text_part = [p for p in result if p["type"] == "text"][0]
        reasoning_part = [p for p in result if p["type"] == "reasoning"][0]
        assert text_part["content"] == "The answer is 42."
        assert reasoning_part["content"] == "I need to calculate..."

    @pytest.mark.asyncio
    async def test_no_recognized_attributes_falls_through_to_str(self):
        part = Mock()
        part.text = None
        part.inline_data = None
        part.mime_type = None
        part.function_call = None
        part.function_response = None
        part.thought = None
        part.__str__ = lambda self: "fallback-string-repr"

        span = _make_span()
        result = await _parts_from_vertex_part_async(part, span, 0)

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["content"] == "fallback-string-repr"
