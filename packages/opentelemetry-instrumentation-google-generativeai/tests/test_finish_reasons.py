"""Unit tests for finish_reason mapping and propagation in span_utils."""

import json
from unittest.mock import MagicMock

import pytest
from opentelemetry.instrumentation.google_generativeai import span_utils as su
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_enum(name):
    """Return a simple object whose ``.name`` attribute equals *name*."""
    obj = MagicMock()
    obj.name = name
    return obj


def _make_candidate(finish_reason=None, text=None):
    """Build a lightweight candidate mock.

    *finish_reason* can be ``None``, a string name (turned into a mock enum),
    or an already-constructed mock enum.
    """
    cand = MagicMock()
    if finish_reason is None:
        cand.finish_reason = None
    elif isinstance(finish_reason, str):
        cand.finish_reason = _make_enum(finish_reason)
    else:
        cand.finish_reason = finish_reason

    # Minimal content with a single text part
    if text is not None:
        part = MagicMock()
        part.text = text
        part.function_call = None
        part.function_response = None
        part.thought = None
        part.inline_data = None
        part.executable_code = None
        part.code_execution_result = None
        cand.content = MagicMock()
        cand.content.parts = [part]
        cand.content.role = "model"
    else:
        cand.content = None
    return cand


def _make_response(candidates=None, text=None):
    """Build a response mock with optional candidates list and fallback text."""
    resp = MagicMock()
    resp.candidates = candidates
    if text is not None:
        resp.text = text
    else:
        # Avoid hasattr returning True for an unwanted .text
        del resp.text
    return resp


def _recorded_output_messages(span_mock):
    """Extract the parsed output-messages JSON from a mock span's set_attribute calls."""
    for call in span_mock.set_attribute.call_args_list:
        args = call[0]
        if args[0] == GenAIAttributes.GEN_AI_OUTPUT_MESSAGES:
            return json.loads(args[1])
    return None


# ===========================================================================
# 1. _map_gemini_finish_reason
# ===========================================================================

class TestMapGeminiFinishReason:
    def test_none_returns_unknown(self):
        assert su._map_gemini_finish_reason(None) == ""

    def test_stop(self):
        assert su._map_gemini_finish_reason(_make_enum("STOP")) == "stop"

    def test_max_tokens(self):
        assert su._map_gemini_finish_reason(_make_enum("MAX_TOKENS")) == "length"

    @pytest.mark.parametrize("name", [
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "LANGUAGE",
    ])
    def test_content_filter_variants(self, name):
        assert su._map_gemini_finish_reason(_make_enum(name)) == "content_filter"

    def test_finish_reason_unspecified(self):
        assert su._map_gemini_finish_reason(_make_enum("FINISH_REASON_UNSPECIFIED")) == ""

    def test_malformed_function_call(self):
        assert su._map_gemini_finish_reason(_make_enum("MALFORMED_FUNCTION_CALL")) == "error"

    def test_other(self):
        assert su._map_gemini_finish_reason(_make_enum("OTHER")) == "error"

    def test_unmapped_enum_returns_unknown(self):
        assert su._map_gemini_finish_reason(_make_enum("TOTALLY_NEW_VALUE")) == ""


# ===========================================================================
# 2. _collect_finish_reasons_from_response
# ===========================================================================

class TestCollectFinishReasonsFromResponse:
    def test_all_none_returns_unknown_entries(self):
        """Keep 1:1 candidate alignment even when finish reasons are unavailable."""
        resp = _make_response(candidates=[_make_candidate(None), _make_candidate(None)])
        assert su._collect_finish_reasons_from_response(resp) == ["", ""]

    def test_meaningful_values_included(self):
        resp = _make_response(candidates=[
            _make_candidate("STOP"),
            _make_candidate("MAX_TOKENS"),
        ])
        assert su._collect_finish_reasons_from_response(resp) == ["stop", "length"]

    def test_mixed_candidates(self):
        """Preserve candidate order and include unknown placeholders."""
        resp = _make_response(candidates=[
            _make_candidate(None),
            _make_candidate("STOP"),
            _make_candidate(None),
        ])
        assert su._collect_finish_reasons_from_response(resp) == ["", "stop", ""]

    def test_no_candidates_attribute(self):
        resp = MagicMock(spec=[])  # no attributes at all
        assert su._collect_finish_reasons_from_response(resp) == []


# ===========================================================================
# 3. Streaming finish reasons (via _collect_finish_reasons_from_response on last chunk)
# ===========================================================================

class TestStreamingFinishReasonsFromLastChunk:
    def test_multiple_candidates_same_reason_preserved(self):
        """Two candidates both with STOP should produce ["stop", "stop"] — no dedup."""
        last_chunk = _make_response(candidates=[
            _make_candidate("STOP", text="a"),
            _make_candidate("STOP", text="b"),
        ])
        reasons = su._collect_finish_reasons_from_response(last_chunk)
        assert reasons == ["stop", "stop"]

    def test_mixed_reasons_preserve_order(self):
        last_chunk = _make_response(candidates=[
            _make_candidate("MAX_TOKENS", text="a"),
            _make_candidate("STOP", text="b"),
        ])
        reasons = su._collect_finish_reasons_from_response(last_chunk)
        assert reasons == ["length", "stop"]

    def test_none_candidates_returns_unknown(self):
        last_chunk = _make_response(candidates=[_make_candidate(None)])
        reasons = su._collect_finish_reasons_from_response(last_chunk)
        assert reasons == [""]


# ===========================================================================
# 4. _output_messages_from_generate_response
# ===========================================================================

class TestOutputMessagesFromGenerateResponse:
    def test_every_message_has_finish_reason_key(self):
        span = MagicMock()
        resp = _make_response(candidates=[
            _make_candidate("STOP", text="hello"),
            _make_candidate(None, text="world"),
        ])
        messages = su._output_messages_from_generate_response(span, resp)
        for msg in messages:
            assert "finish_reason" in msg

    def test_finish_reason_is_always_string(self):
        span = MagicMock()
        resp = _make_response(candidates=[
            _make_candidate(None, text="a"),
            _make_candidate("STOP", text="b"),
        ])
        messages = su._output_messages_from_generate_response(span, resp)
        for msg in messages:
            assert isinstance(msg["finish_reason"], str)
            assert msg["finish_reason"] is not None

    def test_none_finish_reason_becomes_unknown(self):
        span = MagicMock()
        resp = _make_response(candidates=[_make_candidate(None, text="hi")])
        messages = su._output_messages_from_generate_response(span, resp)
        assert messages[0]["finish_reason"] == ""

    def test_stop_finish_reason(self):
        span = MagicMock()
        resp = _make_response(candidates=[_make_candidate("STOP", text="done")])
        messages = su._output_messages_from_generate_response(span, resp)
        assert messages[0]["finish_reason"] == "stop"

    def test_text_fallback_has_unknown_finish_reason(self):
        """When there are no candidates but response.text exists, the fallback message gets 'unknown'."""
        span = MagicMock()
        resp = _make_response(candidates=None, text="fallback text")
        messages = su._output_messages_from_generate_response(span, resp)
        assert len(messages) == 1
        assert messages[0]["finish_reason"] == ""
        assert messages[0]["parts"][0]["content"] == "fallback text"


# ===========================================================================
# 5. set_response_attributes
# ===========================================================================

class TestSetResponseAttributes:
    def test_string_response_has_finish_reason(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = MagicMock()
        span.is_recording.return_value = True

        su.set_response_attributes(span, "Hello world", "gemini-pro")

        messages = _recorded_output_messages(span)
        assert messages is not None
        assert len(messages) == 1
        assert "finish_reason" in messages[0]
        assert messages[0]["finish_reason"] == ""

    def test_string_empty_no_stream_chunk_no_output(self, monkeypatch):
        """Empty string with no stream_last_chunk and unknown finish_reason -> nothing emitted."""
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = MagicMock()
        span.is_recording.return_value = True

        su.set_response_attributes(span, "", "gemini-pro")

        messages = _recorded_output_messages(span)
        assert messages is None  # no set_attribute call for output messages

    def test_string_empty_with_stream_last_chunk_stop(self, monkeypatch):
        """Empty text but stream_last_chunk carries STOP -> message emitted with 'stop'."""
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = MagicMock()
        span.is_recording.return_value = True

        last_chunk = _make_response(candidates=[_make_candidate("STOP")])
        su.set_response_attributes(span, "", "gemini-pro", stream_last_chunk=last_chunk)

        messages = _recorded_output_messages(span)
        assert messages is not None
        assert len(messages) == 1
        assert messages[0]["finish_reason"] == "stop"

    def test_list_of_strings_each_has_finish_reason(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = MagicMock()
        span.is_recording.return_value = True

        su.set_response_attributes(span, ["one", "two", "three"], "gemini-pro")

        messages = _recorded_output_messages(span)
        assert messages is not None
        assert len(messages) == 3
        for msg in messages:
            assert msg["finish_reason"] == ""
            assert msg["role"] == "assistant"


# ===========================================================================
# 5b. set_model_response_attributes finish reason emission
# ===========================================================================

class TestSetModelResponseAttributes:
    def test_omit_top_level_finish_reasons_when_all_unknown(self):
        span = MagicMock()
        span.is_recording.return_value = True

        response = _make_response(candidates=[_make_candidate(None), _make_candidate(None)])
        su.set_model_response_attributes(span, response, "gemini-pro", token_histogram=None)

        finish_reason_calls = [
            c[0]
            for c in span.set_attribute.call_args_list
            if c[0][0] == GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        ]
        assert finish_reason_calls == []

    def test_set_top_level_finish_reasons_with_alignment(self):
        span = MagicMock()
        span.is_recording.return_value = True

        response = _make_response(candidates=[_make_candidate(None), _make_candidate("STOP")])
        su.set_model_response_attributes(span, response, "gemini-pro", token_histogram=None)

        finish_reason_calls = [
            c[0]
            for c in span.set_attribute.call_args_list
            if c[0][0] == GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        ]
        assert len(finish_reason_calls) == 1
        assert finish_reason_calls[0][1] == ["", "stop"]


# ===========================================================================
# 6. _parts_from_genai_part_sync — thinking/reasoning
# ===========================================================================

class TestPartsFromGenaiPartThinking:
    def _make_thinking_part(self, text="Let me think...", thought=True):
        part = MagicMock()
        part.text = text
        part.thought = thought
        part.function_call = None
        part.function_response = None
        part.inline_data = None
        part.executable_code = None
        part.code_execution_result = None
        return part

    def test_thought_true_emits_reasoning_with_text_content(self):
        """When thought=True, emit a single reasoning part with the text content."""
        part = self._make_thinking_part("Let me analyze the problem...")
        span = MagicMock()
        result = su._parts_from_genai_part_sync(part, span, 0)
        assert len(result) == 1
        assert result[0]["type"] == "reasoning"
        assert result[0]["content"] == "Let me analyze the problem..."

    def test_thought_false_emits_text(self):
        """When thought=False, emit a normal text part."""
        part = self._make_thinking_part("Hello world", thought=False)
        result = su._parts_from_genai_part_sync(part, MagicMock(), 0)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["content"] == "Hello world"

    def test_thought_none_emits_text(self):
        """When thought=None, emit a normal text part."""
        part = self._make_thinking_part("Hello world", thought=None)
        result = su._parts_from_genai_part_sync(part, MagicMock(), 0)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["content"] == "Hello world"

    def test_no_duplicate_parts_for_thinking(self):
        """Must NOT produce both a text and reasoning part for the same thinking block."""
        part = self._make_thinking_part("Deep thoughts here")
        result = su._parts_from_genai_part_sync(part, MagicMock(), 0)
        types_emitted = [r["type"] for r in result]
        assert types_emitted.count("text") == 0
        assert types_emitted.count("reasoning") == 1
