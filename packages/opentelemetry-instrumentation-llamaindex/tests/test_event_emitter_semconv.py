"""Unit tests for event_emitter semconv migration."""

from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.llamaindex.event_emitter import (
    EVENT_ATTRIBUTES,
    emit_chat_response_events,
)
from opentelemetry.instrumentation.llamaindex.event_models import ChoiceEvent


# ===========================================================================
# EVENT_ATTRIBUTES — provider name
# ===========================================================================

class TestEventAttributes:
    def test_uses_gen_ai_provider_name(self):
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in EVENT_ATTRIBUTES
        assert GenAIAttributes.GEN_AI_SYSTEM not in EVENT_ATTRIBUTES

    def test_value_is_llamaindex(self):
        assert EVENT_ATTRIBUTES[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "llamaindex"


# ===========================================================================
# ChoiceEvent — default finish_reason
# ===========================================================================

class TestChoiceEventDefault:
    def test_default_finish_reason_is_empty_string(self):
        event = ChoiceEvent(index=0, message={"content": "ok", "role": "assistant"})
        assert event.finish_reason == ""


# ===========================================================================
# emit_chat_response_events — finish_reason
# ===========================================================================

class TestEmitChatResponseEvents:
    def test_finish_reason_tool_calls_passed_through(self):
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = {"choices": [{"finish_reason": "tool_calls"}]}
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == "tool_calls"

    def test_finish_reason_fallback_empty_string_for_events(self):
        """Event emission path uses '' fallback (ChoiceEvent convention)."""
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = {}  # no choices
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == ""

    def test_finish_reason_handles_none_raw(self):
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = None
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == ""

    def test_cohere_finish_reason_mapped(self):
        """Cohere-style finish reasons should be mapped (not just OpenAI format)."""
        from types import SimpleNamespace
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = SimpleNamespace(finish_reason="COMPLETE")
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == "stop"

    def test_anthropic_finish_reason_mapped(self):
        """Anthropic-style stop_reason should be mapped."""
        from types import SimpleNamespace
        event = MagicMock()
        event.response = MagicMock()
        event.response.raw = SimpleNamespace(stop_reason="end_turn")
        event.response.message.content = "hi"
        event.response.message.role.value = "assistant"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_choice_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_response_events(event)
            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["finish_reason"] == "stop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
