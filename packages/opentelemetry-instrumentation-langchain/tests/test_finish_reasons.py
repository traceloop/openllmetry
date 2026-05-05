"""
Tests for gen_ai.response.finish_reasons span attribute.

Covers:
- Mapped finish reasons (OpenAI & Anthropic native values)
- None / missing finish reason (attribute must be omitted)
- Finish reasons recorded even when should_send_prompts() is False
"""

import json

import pytest
from unittest.mock import Mock
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_core.messages import AIMessage
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.instrumentation.langchain.span_utils import set_chat_response, _map_finish_reason


@pytest.fixture
def mock_span():
    span = Mock()
    span.is_recording.return_value = True
    span.attributes = {}

    def set_attribute(key, value):
        span.attributes[key] = value

    span.set_attribute = set_attribute
    return span


# ---------- _map_finish_reason unit tests ----------

class TestMapFinishReason:
    @pytest.mark.parametrize("raw,expected", [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_call"),
        ("function_call", "tool_call"),
        ("content_filter", "content_filter"),
        # Anthropic
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("tool_use", "tool_call"),
        ("max_tokens", "length"),
        # Unknown passthrough
        ("some_future_reason", "some_future_reason"),
    ])
    def test_known_and_unknown_reasons(self, raw, expected):
        assert _map_finish_reason(raw) == expected

    def test_none_returns_none(self):
        assert _map_finish_reason(None) is None

    def test_empty_string_returns_empty(self):
        assert _map_finish_reason("") == ""


# ---------- span-level finish_reasons tests ----------

class TestFinishReasonsSpanAttribute:
    def _make_generation(self, content="OK", finish_reason=None):
        gen_info = {"finish_reason": finish_reason} if finish_reason else {}
        return ChatGeneration(
            message=AIMessage(content=content),
            generation_info=gen_info,
        )

    def test_stop_reason_mapped(self, mock_span, monkeypatch):
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True,
        )
        gen = self._make_generation(finish_reason="stop")
        set_chat_response(mock_span, LLMResult(generations=[[gen]]))

        assert mock_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]

        output = json.loads(mock_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert output[0]["finish_reason"] == "stop"

    def test_anthropic_end_turn_mapped_to_stop(self, mock_span, monkeypatch):
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True,
        )
        gen = self._make_generation(finish_reason="end_turn")
        set_chat_response(mock_span, LLMResult(generations=[[gen]]))

        assert mock_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]

    def test_tool_calls_mapped(self, mock_span, monkeypatch):
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True,
        )
        gen = self._make_generation(finish_reason="tool_calls")
        set_chat_response(mock_span, LLMResult(generations=[[gen]]))

        assert mock_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["tool_call"]

    def test_no_finish_reason_omits_attribute(self, mock_span, monkeypatch):
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True,
        )
        gen = self._make_generation(finish_reason=None)
        set_chat_response(mock_span, LLMResult(generations=[[gen]]))

        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in mock_span.attributes

        output = json.loads(mock_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert output[0]["finish_reason"] == ""

    def test_empty_generation_info_omits_attribute(self, mock_span, monkeypatch):
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True,
        )
        gen = ChatGeneration(
            message=AIMessage(content="hi"),
            generation_info={},
        )
        set_chat_response(mock_span, LLMResult(generations=[[gen]]))

        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in mock_span.attributes

    def test_finish_reasons_recorded_when_prompts_disabled(self, mock_span, monkeypatch):
        """Finish reasons are metadata, not content — must be recorded even when
        should_send_prompts() returns False."""
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: False,
        )
        gen = self._make_generation(finish_reason="stop")
        set_chat_response(mock_span, LLMResult(generations=[[gen]]))

        # finish_reasons MUST still be recorded
        assert mock_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]

        # output messages must NOT be recorded (content gated)
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in mock_span.attributes

    def test_deduplicated_finish_reasons(self, mock_span, monkeypatch):
        """Multiple generations with the same reason should not duplicate."""
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True,
        )
        gen1 = self._make_generation(content="a", finish_reason="stop")
        gen2 = self._make_generation(content="b", finish_reason="stop")
        set_chat_response(mock_span, LLMResult(generations=[[gen1, gen2]]))

        assert mock_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]

    def test_multiple_distinct_finish_reasons(self, mock_span, monkeypatch):
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True,
        )
        gen1 = self._make_generation(content="a", finish_reason="stop")
        gen2 = self._make_generation(content="b", finish_reason="tool_calls")
        set_chat_response(mock_span, LLMResult(generations=[[gen1], [gen2]]))

        reasons = mock_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
        assert "stop" in reasons
        assert "tool_call" in reasons
        assert len(reasons) == 2
