"""Unit tests for event_emitter semconv migration."""

from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.llamaindex.event_emitter import (
    _event_attributes,
    emit_chat_message_events,
    emit_chat_response_events,
)
from opentelemetry.instrumentation.llamaindex.event_models import ChoiceEvent


# ===========================================================================
# _event_attributes — dynamic provider name
# ===========================================================================

class TestEventAttributes:
    def test_with_provider_name(self):
        attrs = _event_attributes("openai")
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"

    def test_with_none_provider_name(self):
        attrs = _event_attributes(None)
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in attrs

    def test_with_empty_provider_name(self):
        attrs = _event_attributes("")
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in attrs

    def test_uses_actual_provider_not_llamaindex(self):
        """Provider name should be the actual LLM provider, not the framework."""
        attrs = _event_attributes("anthropic")
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "anthropic"


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

class TestEmitChatMessageEventsProviderName:
    def test_provider_name_passed_to_message_events(self):
        """emit_chat_message_events should forward provider_name to emit_event."""
        event = MagicMock()
        msg = MagicMock()
        msg.content = "hello"
        msg.role.value = "user"
        event.messages = [msg]

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_message_events(event, provider_name="openai")
            mock_emit.assert_called_once()
            assert mock_emit.call_args[1]["provider_name"] == "openai"

    def test_none_provider_name_passed_through(self):
        event = MagicMock()
        msg = MagicMock()
        msg.content = "hello"
        msg.role.value = "user"
        event.messages = [msg]

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_chat_message_events(event)
            mock_emit.assert_called_once()
            assert mock_emit.call_args[1]["provider_name"] is None

    def test_backward_compat_no_provider_arg(self):
        """Calling without provider_name should still work (defaults to None)."""
        event = MagicMock()
        event.messages = []
        # Should not raise
        emit_chat_message_events(event)


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
