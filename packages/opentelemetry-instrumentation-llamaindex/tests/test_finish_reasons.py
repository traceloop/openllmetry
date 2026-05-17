"""Dedicated finish_reason tests — mapping, extraction, span attributes, output messages, events.

Groups:
- map_finish_reason: provider value → OTel enum mapping
- extract_finish_reasons: raw response → List[str] extraction per provider format
- Span attribute: gen_ai.response.finish_reasons through span_utils and custom_llm_instrumentor
- Output messages: finish_reason key in gen_ai.output.messages JSON
- Events: finish_reason in ChoiceEvent emission
- Cross-cutting invariants: always string, never empty array, always present key
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.instrumentation.llamaindex._message_utils import (
    build_completion_output_message,
    build_output_message,
    map_finish_reason,
)
from opentelemetry.instrumentation.llamaindex._response_utils import (
    extract_finish_reasons,
)
from opentelemetry.instrumentation.llamaindex.custom_llm_instrumentor import (
    _handle_response,
)
from opentelemetry.instrumentation.llamaindex.event_emitter import (
    emit_chat_response_events,
)
from opentelemetry.instrumentation.llamaindex.event_models import ChoiceEvent
from opentelemetry.instrumentation.llamaindex.span_utils import (
    set_llm_chat_response,
    set_llm_chat_response_model_attributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


VALID_OTEL_FINISH_REASONS = {"stop", "tool_call", "length", "content_filter", "error"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role, content, **additional_kwargs):
    """Create a fake ChatMessage-like object."""
    return SimpleNamespace(
        role=SimpleNamespace(value=role),
        content=content,
        additional_kwargs=additional_kwargs,
    )


def _recording_span():
    span = MagicMock()
    span.is_recording.return_value = True
    return span


def _attr(span, name):
    for call in span.set_attribute.call_args_list:
        if call.args[0] == name:
            return call.args[1]
    return None


def _has_attr(span, name):
    return any(c.args[0] == name for c in span.set_attribute.call_args_list)


def _custom_llm_instance(class_name="Ollama", model_name="llama3"):
    inst = type(class_name, (), {})()
    inst.metadata = SimpleNamespace(
        model_name=model_name, context_window=4096, num_output=512
    )
    return inst


PATCH_SHOULD_SEND_SPAN = "opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts"
PATCH_SHOULD_SEND_CUSTOM = "opentelemetry.instrumentation.llamaindex.custom_llm_instrumentor.should_send_prompts"


# ===========================================================================
# map_finish_reason — provider value → OTel enum
# ===========================================================================


class TestMapFinishReason:
    # --- OpenAI ---
    def test_openai_stop(self):
        assert map_finish_reason("stop") == "stop"

    def test_openai_tool_calls_mapped_to_singular(self):
        """OTel JSON schema uses 'tool_call' (singular). OpenAI 'tool_calls' (plural) must be mapped."""
        assert map_finish_reason("tool_calls") == "tool_call"

    def test_openai_function_call(self):
        assert map_finish_reason("function_call") == "tool_call"

    def test_openai_length(self):
        assert map_finish_reason("length") == "length"

    def test_openai_content_filter(self):
        assert map_finish_reason("content_filter") == "content_filter"

    # --- Cohere ---
    def test_cohere_complete(self):
        assert map_finish_reason("COMPLETE") == "stop"

    def test_cohere_max_tokens(self):
        assert map_finish_reason("MAX_TOKENS") == "length"

    def test_cohere_error(self):
        assert map_finish_reason("ERROR") == "error"

    def test_cohere_error_toxic(self):
        assert map_finish_reason("ERROR_TOXIC") == "content_filter"

    # --- Anthropic ---
    def test_anthropic_end_turn(self):
        assert map_finish_reason("end_turn") == "stop"

    def test_anthropic_stop_sequence(self):
        assert map_finish_reason("stop_sequence") == "stop"

    def test_anthropic_tool_use(self):
        assert map_finish_reason("tool_use") == "tool_call"

    def test_anthropic_max_tokens(self):
        assert map_finish_reason("max_tokens") == "length"

    # --- Google Gemini ---
    def test_gemini_stop(self):
        assert map_finish_reason("STOP") == "stop"

    def test_gemini_safety(self):
        assert map_finish_reason("SAFETY") == "content_filter"

    def test_gemini_recitation(self):
        assert map_finish_reason("RECITATION") == "content_filter"

    def test_gemini_blocklist(self):
        assert map_finish_reason("BLOCKLIST") == "content_filter"

    def test_gemini_prohibited_content(self):
        assert map_finish_reason("PROHIBITED_CONTENT") == "content_filter"

    def test_gemini_spii(self):
        assert map_finish_reason("SPII") == "content_filter"

    def test_gemini_unspecified(self):
        assert map_finish_reason("FINISH_REASON_UNSPECIFIED") == "error"

    def test_gemini_other(self):
        assert map_finish_reason("OTHER") == "error"

    def test_gemini_max_tokens_reuses_cohere_mapping(self):
        """Gemini MAX_TOKENS is handled by the same mapping as Cohere."""
        assert map_finish_reason("MAX_TOKENS") == "length"

    # --- Edge cases ---
    def test_none_returns_none(self):
        assert map_finish_reason(None) is None

    def test_empty_string_returns_none(self):
        assert map_finish_reason("") is None

    def test_unknown_passes_through(self):
        assert map_finish_reason("custom_reason") == "custom_reason"

    def test_all_mapped_values_are_valid_otel(self):
        """Every value in _FINISH_REASON_MAP must produce a valid OTel finish reason."""
        from opentelemetry.instrumentation.llamaindex._message_utils import _FINISH_REASON_MAP
        for provider_val, otel_val in _FINISH_REASON_MAP.items():
            assert otel_val in VALID_OTEL_FINISH_REASONS, (
                f"Mapping '{provider_val}' -> '{otel_val}' is not a valid OTel finish reason"
            )


# ===========================================================================
# extract_finish_reasons — raw response → List[str]
# ===========================================================================


class TestExtractFinishReasons:
    # --- OpenAI choices[] ---
    def test_openai_choices_object(self):
        choice = SimpleNamespace(finish_reason="stop")
        raw = SimpleNamespace(choices=[choice])
        assert extract_finish_reasons(raw) == ["stop"]

    def test_openai_choices_dict(self):
        raw = {"choices": [{"finish_reason": "stop"}]}
        assert extract_finish_reasons(raw) == ["stop"]

    def test_openai_tool_calls_mapped_to_singular(self):
        raw = {"choices": [{"finish_reason": "tool_calls"}]}
        assert extract_finish_reasons(raw) == ["tool_call"]

    def test_multiple_choices(self):
        raw = {"choices": [{"finish_reason": "stop"}, {"finish_reason": "length"}]}
        assert extract_finish_reasons(raw) == ["stop", "length"]

    def test_none_finish_reason_in_choices(self):
        raw = {"choices": [{"finish_reason": None}]}
        assert extract_finish_reasons(raw) == []

    def test_empty_choices(self):
        raw = {"choices": []}
        assert extract_finish_reasons(raw) == []

    # --- Anthropic stop_reason ---
    def test_anthropic_stop_reason(self):
        raw = SimpleNamespace(stop_reason="end_turn")
        assert extract_finish_reasons(raw) == ["stop"]

    def test_anthropic_stop_reason_dict(self):
        raw = {"stop_reason": "end_turn"}
        assert extract_finish_reasons(raw) == ["stop"]

    # --- Cohere finish_reason ---
    def test_cohere_finish_reason(self):
        raw = SimpleNamespace(finish_reason="COMPLETE")
        assert extract_finish_reasons(raw) == ["stop"]

    def test_cohere_finish_reason_dict(self):
        raw = {"finish_reason": "MAX_TOKENS"}
        assert extract_finish_reasons(raw) == ["length"]

    # --- Google Gemini candidates[] ---
    def test_gemini_candidates_object(self):
        candidate = SimpleNamespace(finish_reason="STOP")
        raw = SimpleNamespace(candidates=[candidate])
        assert extract_finish_reasons(raw) == ["stop"]

    def test_gemini_candidates_dict(self):
        raw = {"candidates": [{"finish_reason": "STOP"}]}
        assert extract_finish_reasons(raw) == ["stop"]

    def test_gemini_candidates_max_tokens(self):
        candidate = SimpleNamespace(finish_reason="MAX_TOKENS")
        raw = SimpleNamespace(candidates=[candidate])
        assert extract_finish_reasons(raw) == ["length"]

    def test_gemini_candidates_safety(self):
        candidate = SimpleNamespace(finish_reason="SAFETY")
        raw = SimpleNamespace(candidates=[candidate])
        assert extract_finish_reasons(raw) == ["content_filter"]

    def test_gemini_candidates_enum_object(self):
        """Gemini finish_reason may be an enum with a .name attribute."""
        enum_val = SimpleNamespace(name="STOP")
        candidate = SimpleNamespace(finish_reason=enum_val)
        raw = SimpleNamespace(candidates=[candidate])
        assert extract_finish_reasons(raw) == ["stop"]

    def test_gemini_candidates_enum_unspecified(self):
        enum_val = SimpleNamespace(name="FINISH_REASON_UNSPECIFIED")
        candidate = SimpleNamespace(finish_reason=enum_val)
        raw = SimpleNamespace(candidates=[candidate])
        assert extract_finish_reasons(raw) == ["error"]

    def test_gemini_candidates_none_finish_reason(self):
        candidate = SimpleNamespace(finish_reason=None)
        raw = SimpleNamespace(candidates=[candidate])
        assert extract_finish_reasons(raw) == []

    def test_gemini_multiple_candidates(self):
        raw = {"candidates": [{"finish_reason": "STOP"}, {"finish_reason": "SAFETY"}]}
        assert extract_finish_reasons(raw) == ["stop", "content_filter"]

    def test_gemini_empty_candidates(self):
        raw = {"candidates": []}
        assert extract_finish_reasons(raw) == []

    # --- Ollama done_reason ---
    def test_ollama_done_reason(self):
        raw = {"done_reason": "stop"}
        assert extract_finish_reasons(raw) == ["stop"]

    def test_ollama_done_reason_object(self):
        raw = SimpleNamespace(done_reason="stop")
        assert extract_finish_reasons(raw) == ["stop"]

    def test_ollama_done_reason_length(self):
        raw = {"done_reason": "length"}
        assert extract_finish_reasons(raw) == ["length"]

    def test_ollama_done_reason_none(self):
        raw = {"done_reason": None}
        assert extract_finish_reasons(raw) == []

    # --- Priority / precedence ---
    def test_choices_takes_precedence_over_candidates(self):
        raw = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop")],
            candidates=[SimpleNamespace(finish_reason="SAFETY")],
        )
        assert extract_finish_reasons(raw) == ["stop"]

    def test_candidates_takes_precedence_over_stop_reason(self):
        raw = SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason="STOP")],
            stop_reason="end_turn",
        )
        assert extract_finish_reasons(raw) == ["stop"]

    def test_stop_reason_takes_precedence_over_done_reason(self):
        raw = {"stop_reason": "end_turn", "done_reason": "length"}
        assert extract_finish_reasons(raw) == ["stop"]

    # --- Edge cases ---
    def test_none_raw(self):
        assert extract_finish_reasons(None) == []

    def test_no_finish_reason(self):
        assert extract_finish_reasons(SimpleNamespace()) == []

    def test_empty_dict_raw(self):
        assert extract_finish_reasons({}) == []


# ===========================================================================
# Span attribute: gen_ai.response.finish_reasons
# ===========================================================================


class TestSpanFinishReasonsViaSpanUtils:
    """Tests for gen_ai.response.finish_reasons set by set_llm_chat_response_model_attributes."""

    def _event_with_raw(self, raw):
        event = MagicMock()
        event.response = MagicMock(raw=raw)
        return event

    def test_sets_finish_reasons_array(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="gpt-4",
            choices=[SimpleNamespace(finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_finish_reasons_tool_calls_mapped_to_singular(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="gpt-4",
            choices=[SimpleNamespace(finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["tool_call"]

    def test_finish_reasons_omitted_when_not_available(self):
        span = _recording_span()
        raw = SimpleNamespace(model="gpt-4")
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)

    def test_finish_reasons_not_gated_by_should_send_prompts(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="gpt-4",
            choices=[SimpleNamespace(finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        with patch(PATCH_SHOULD_SEND_SPAN, return_value=False):
            set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_no_legacy_finish_reason_attr(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="gpt-4",
            choices=[SimpleNamespace(finish_reason="stop")],
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert not _has_attr(span, SpanAttributes.LLM_RESPONSE_FINISH_REASON)

    def test_cohere_finish_reason_mapped(self):
        span = _recording_span()
        raw = SimpleNamespace(model="command-r", finish_reason="COMPLETE")
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_gemini_candidates_through_span_utils(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="gemini-pro",
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_gemini_safety_through_span_utils(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="gemini-pro",
            candidates=[SimpleNamespace(finish_reason="SAFETY")],
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["content_filter"]

    def test_gemini_enum_object_through_span_utils(self):
        """Gemini finish_reason may be a protobuf enum object, not a string."""
        span = _recording_span()
        enum_val = SimpleNamespace(name="STOP")
        raw = SimpleNamespace(
            model="gemini-pro",
            candidates=[SimpleNamespace(finish_reason=enum_val)],
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_ollama_done_reason_through_span_utils(self):
        span = _recording_span()
        raw = SimpleNamespace(model="llama3", done_reason="stop")
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_none_raw_omits_finish_reasons(self):
        span = _recording_span()
        event = MagicMock()
        event.response = MagicMock(raw=None)
        set_llm_chat_response_model_attributes(event, span)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)

    def test_multiple_choices_mixed_reasons(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="gpt-4",
            choices=[
                SimpleNamespace(finish_reason="stop"),
                SimpleNamespace(finish_reason="tool_calls"),
                SimpleNamespace(finish_reason="length"),
            ],
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == [
            "stop", "tool_call", "length"
        ]


class TestSpanFinishReasonsViaCustomLLM:
    """Tests for gen_ai.response.finish_reasons set by custom_llm_instrumentor._handle_response."""

    def test_sets_finish_reasons_from_raw(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _recording_span()
        inst = _custom_llm_instance()
        resp = SimpleNamespace(text="ok", raw={"choices": [{"finish_reason": "stop"}]})
        _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_finish_reasons_not_gated(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _recording_span()
        inst = _custom_llm_instance()
        resp = SimpleNamespace(text="ok", raw={"choices": [{"finish_reason": "stop"}]})
        with patch(PATCH_SHOULD_SEND_CUSTOM, return_value=False):
            _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_gemini_candidates_through_custom_llm(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _recording_span()
        inst = _custom_llm_instance("Gemini", "gemini-pro")
        raw = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])
        resp = SimpleNamespace(text="ok", raw=raw)
        _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_ollama_done_reason_through_custom_llm(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _recording_span()
        inst = _custom_llm_instance()
        resp = SimpleNamespace(text="ok", raw=SimpleNamespace(done_reason="stop"))
        _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_no_raw_omits_finish_reasons(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _recording_span()
        inst = _custom_llm_instance()
        resp = SimpleNamespace(text="ok", raw=None)
        _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)


# ===========================================================================
# Output messages: finish_reason in gen_ai.output.messages JSON
# ===========================================================================


class TestOutputMessageFinishReason:
    """Tests for finish_reason field inside output message JSON."""

    # --- build_output_message ---
    def test_stop(self):
        resp = _msg("assistant", "Done.")
        result = build_output_message(resp, finish_reason="stop")
        assert result["finish_reason"] == "stop"

    def test_tool_calls_mapped_to_singular(self):
        resp = _msg("assistant", "")
        result = build_output_message(resp, finish_reason="tool_calls")
        assert result["finish_reason"] == "tool_call"

    def test_length(self):
        resp = _msg("assistant", "truncated...")
        result = build_output_message(resp, finish_reason="length")
        assert result["finish_reason"] == "length"

    def test_none_defaults_to_empty_string(self):
        """Missing finish_reason defaults to '' (required field, non-nullable per schema)."""
        resp = _msg("assistant", "ok")
        result = build_output_message(resp, finish_reason=None)
        assert result["finish_reason"] == ""

    def test_unknown_passes_through(self):
        resp = _msg("assistant", "ok")
        result = build_output_message(resp, finish_reason="custom_stop")
        assert result["finish_reason"] == "custom_stop"

    def test_always_string_type(self):
        resp = _msg("assistant", "ok")
        result = build_output_message(resp)
        assert isinstance(result["finish_reason"], str)

    # --- build_completion_output_message ---
    def test_completion_with_finish_reason(self):
        result = build_completion_output_message("done", finish_reason="stop")
        assert result["finish_reason"] == "stop"

    def test_completion_with_mapped_finish_reason(self):
        result = build_completion_output_message("done", finish_reason="COMPLETE")
        assert result["finish_reason"] == "stop"

    def test_completion_none_defaults_to_empty_string(self):
        result = build_completion_output_message("done", finish_reason=None)
        assert result["finish_reason"] == ""

    # --- Through span_utils ---
    def test_output_message_includes_finish_reason_via_span_utils(self):
        span = _recording_span()
        msg = _msg("assistant", "The answer is 42.")
        event = MagicMock()
        event.response = MagicMock(
            message=msg,
            raw={"choices": [{"finish_reason": "stop"}]},
        )
        with patch(PATCH_SHOULD_SEND_SPAN, return_value=True):
            set_llm_chat_response(event, span)
        msgs = json.loads(_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert msgs[0]["finish_reason"] == "stop"

    def test_output_message_finish_reason_empty_when_no_raw(self):
        span = _recording_span()
        msg = _msg("assistant", "ok")
        event = MagicMock()
        event.response = MagicMock(message=msg, raw=None)
        with patch(PATCH_SHOULD_SEND_SPAN, return_value=True):
            set_llm_chat_response(event, span)
        msgs = json.loads(_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert msgs[0]["finish_reason"] == ""

    def test_output_message_gemini_finish_reason_via_span_utils(self):
        span = _recording_span()
        msg = _msg("assistant", "ok")
        event = MagicMock()
        event.response = MagicMock(
            message=msg,
            raw=SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")]),
        )
        with patch(PATCH_SHOULD_SEND_SPAN, return_value=True):
            set_llm_chat_response(event, span)
        msgs = json.loads(_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert msgs[0]["finish_reason"] == "stop"

    # --- Through custom_llm_instrumentor ---
    def test_output_message_finish_reason_via_custom_llm_chat(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _recording_span()
        inst = _custom_llm_instance()
        msg = _msg("assistant", "Reply")
        resp = SimpleNamespace(
            text="Reply",
            message=msg,
            raw={"choices": [{"finish_reason": "stop"}]},
        )
        with patch(PATCH_SHOULD_SEND_CUSTOM, return_value=True):
            _handle_response(span, LLMRequestTypeValues.CHAT, inst, resp)
        msgs = json.loads(_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert msgs[0]["finish_reason"] == "stop"

    def test_output_message_finish_reason_via_custom_llm_completion(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _recording_span()
        inst = _custom_llm_instance()
        resp = SimpleNamespace(text="done", raw={"choices": [{"finish_reason": "length"}]})
        with patch(PATCH_SHOULD_SEND_CUSTOM, return_value=True):
            _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        msgs = json.loads(_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert msgs[0]["finish_reason"] == "length"


# ===========================================================================
# Events: finish_reason in ChoiceEvent emission
# ===========================================================================


class TestChoiceEventFinishReason:
    def test_default_finish_reason_is_empty_string(self):
        event = ChoiceEvent(index=0, message={"content": "ok", "role": "assistant"})
        assert event.finish_reason == ""


class TestEmitChatResponseEventsFinishReason:
    def test_tool_calls_mapped_to_singular(self):
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = {"choices": [{"finish_reason": "tool_calls"}]}
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == "tool_call"

    def test_fallback_empty_string_when_no_raw_reasons(self):
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = {}
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == ""

    def test_handles_none_raw(self):
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = None
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == ""

    def test_provider_name_passed_to_choice_event(self):
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = {"choices": [{"finish_reason": "stop"}]}
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event, provider_name="anthropic")
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["provider_name"] == "anthropic"

    def test_cohere_finish_reason_mapped(self):
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = SimpleNamespace(finish_reason="COMPLETE")
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == "stop"

    def test_anthropic_finish_reason_mapped(self):
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = SimpleNamespace(stop_reason="end_turn")
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == "stop"

    def test_gemini_finish_reason_mapped(self):
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == "stop"


# ===========================================================================
# Cross-cutting invariants
# ===========================================================================


class TestFinishReasonInvariants:
    """Invariants that must hold across all code paths."""

    def test_output_message_finish_reason_key_always_present(self):
        """finish_reason is required per OutputMessage JSON schema — key must always exist."""
        resp = _msg("assistant", "ok")
        for fr in [None, "", "stop", "length", "tool_calls", "custom"]:
            result = build_output_message(resp, finish_reason=fr)
            assert "finish_reason" in result, f"finish_reason key missing for input {fr!r}"

    def test_output_message_finish_reason_never_none(self):
        """finish_reason must always be a string, never None (schema is non-nullable)."""
        resp = _msg("assistant", "ok")
        for fr in [None, "", "stop"]:
            result = build_output_message(resp, finish_reason=fr)
            assert result["finish_reason"] is not None, f"finish_reason is None for input {fr!r}"
            assert isinstance(result["finish_reason"], str), f"finish_reason not str for input {fr!r}"

    def test_completion_output_message_finish_reason_never_none(self):
        for fr in [None, "", "stop"]:
            result = build_completion_output_message("text", finish_reason=fr)
            assert result["finish_reason"] is not None
            assert isinstance(result["finish_reason"], str)

    def test_span_attr_never_set_as_empty_array(self):
        """gen_ai.response.finish_reasons must be omitted, not set as []."""
        span = _recording_span()
        # Raw with no finish_reason at all
        raw = SimpleNamespace(model="gpt-4")
        event = MagicMock()
        event.response = MagicMock(raw=raw)
        set_llm_chat_response_model_attributes(event, span)

        # Attribute should not be set at all — never as []
        fr = _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert fr is None, "finish_reasons must be omitted when unavailable, not set as empty"

    def test_span_attr_never_set_as_empty_array_custom_llm(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _recording_span()
        inst = _custom_llm_instance()
        resp = SimpleNamespace(text="ok", raw=SimpleNamespace(model="llama3"))
        _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        fr = _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert fr is None, "finish_reasons must be omitted when unavailable, not set as empty"

    def test_all_otel_enum_values_pass_through_mapper(self):
        """The five canonical OTel values should pass through unchanged."""
        for val in VALID_OTEL_FINISH_REASONS:
            assert map_finish_reason(val) == val, f"OTel value '{val}' was altered by mapper"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
