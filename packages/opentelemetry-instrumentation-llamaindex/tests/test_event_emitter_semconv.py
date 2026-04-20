"""Unit tests for event_emitter semconv migration."""

from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.llamaindex.event_emitter import (
    _event_attributes,
    emit_chat_message_events,
    emit_rerank_message_event,
)


# ===========================================================================
# _event_attributes — dynamic provider name
# ===========================================================================

class TestEventAttributes:
    def test_with_provider_name(self):
        attrs = _event_attributes("openai")
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"

    def test_with_none_provider_name_falls_back_to_llamaindex(self):
        """When provider is unknown, fall back to 'llamaindex' so events always have a provider."""
        attrs = _event_attributes(None)
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "llamaindex"

    def test_with_empty_provider_name_falls_back_to_llamaindex(self):
        attrs = _event_attributes("")
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "llamaindex"

    def test_uses_actual_provider_not_llamaindex(self):
        """Provider name should be the actual LLM provider, not the framework."""
        attrs = _event_attributes("anthropic")
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "anthropic"


# ===========================================================================
# emit_chat_message_events — provider_name
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


# ===========================================================================
# emit_rerank_message_event — provider_name
# ===========================================================================

class TestEmitRerankMessageEvent:
    def test_provider_name_passed_to_rerank_event(self):
        """emit_rerank_message_event should forward provider_name."""
        event = MagicMock()
        event.query = "search query"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_rerank_message_event(event, provider_name="cohere")
            mock_emit.assert_called_once()
            assert mock_emit.call_args[1]["provider_name"] == "cohere"

    def test_rerank_without_provider_name(self):
        """Rerank events without provider_name should still work."""
        event = MagicMock()
        event.query = "search query"

        with patch("opentelemetry.instrumentation.llamaindex.event_emitter.emit_event") as mock_emit:
            with patch("opentelemetry.instrumentation.llamaindex.event_emitter.should_emit_events", return_value=True):
                emit_rerank_message_event(event)
            mock_emit.assert_called_once()
            assert mock_emit.call_args[1]["provider_name"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
