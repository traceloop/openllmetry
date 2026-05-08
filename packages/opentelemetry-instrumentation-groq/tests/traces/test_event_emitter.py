"""
Unit tests for event_emitter helpers.

Covers:
  emit_event, _emit_message_event, _emit_choice_event.
All tests are pure unit tests — no network calls, no cassettes.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.instrumentation.groq.event_emitter import (
    _emit_choice_event,
    _emit_message_event,
    emit_event,
    emit_message_events,
)
from opentelemetry.instrumentation.groq.event_models import ChoiceEvent, MessageEvent
from opentelemetry.instrumentation.groq.utils import TRACELOOP_TRACE_CONTENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logger():
    logger = MagicMock()
    logger.emit = MagicMock()
    return logger


def _emitted_body(logger):
    """Return the body of the first emitted LogRecord."""
    return logger.emit.call_args[0][0].body


# ---------------------------------------------------------------------------
# emit_event
# ---------------------------------------------------------------------------


class TestEmitEvent:
    def test_none_logger_returns_without_emitting(self):
        # Should not raise; nothing emitted
        emit_event(MessageEvent(content="hello", role="user"), event_logger=None)

    def test_events_disabled_returns_without_emitting(self):
        logger = _logger()
        with patch(
            "opentelemetry.instrumentation.groq.event_emitter.should_emit_events",
            return_value=False,
        ):
            emit_event(MessageEvent(content="hello", role="user"), event_logger=logger)
        logger.emit.assert_not_called()

    def test_unsupported_event_type_raises_type_error(self):
        logger = _logger()
        with patch(
            "opentelemetry.instrumentation.groq.event_emitter.should_emit_events",
            return_value=True,
        ):
            with pytest.raises(TypeError, match="Unsupported event type"):
                emit_event("not_an_event_object", event_logger=logger)


# ---------------------------------------------------------------------------
# emit_message_events
# ---------------------------------------------------------------------------


class TestEmitMessageEvents:
    def test_tool_calls_forwarded_from_assistant_message(self):
        """emit_message_events must pass tool_calls from kwargs to MessageEvent."""
        logger = _logger()
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        kwargs = {
            "messages": [{"role": "assistant", "content": None, "tool_calls": tool_calls}]
        }
        with patch(
            "opentelemetry.instrumentation.groq.event_emitter.should_emit_events",
            return_value=True,
        ):
            emit_message_events(kwargs, logger)
        body = logger.emit.call_args[0][0].body
        assert "tool_calls" in body
        assert body["tool_calls"][0]["id"] == "c1"

    def test_messages_without_tool_calls_do_not_include_tool_calls_key(self):
        logger = _logger()
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        with patch(
            "opentelemetry.instrumentation.groq.event_emitter.should_emit_events",
            return_value=True,
        ):
            emit_message_events(kwargs, logger)
        body = logger.emit.call_args[0][0].body
        assert "tool_calls" not in body


# ---------------------------------------------------------------------------
# _emit_message_event
# ---------------------------------------------------------------------------


class TestEmitMessageEvent:
    def test_invalid_role_uses_fallback_event_name(self):
        logger = _logger()
        event = MessageEvent(content="hi", role="some_unknown_role")
        _emit_message_event(event, logger)
        log_record = logger.emit.call_args[0][0]
        assert log_record.event_name == "gen_ai.user.message"

    def test_non_assistant_role_with_tool_calls_removes_tool_calls(self):
        logger = _logger()
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        event = MessageEvent(content="some", role="user", tool_calls=tool_calls)
        _emit_message_event(event, logger)
        body = _emitted_body(logger)
        assert "tool_calls" not in body

    def test_assistant_role_with_tool_calls_keeps_tool_calls(self):
        logger = _logger()
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        event = MessageEvent(content="", role="assistant", tool_calls=tool_calls)
        _emit_message_event(event, logger)
        body = _emitted_body(logger)
        assert "tool_calls" in body

    def test_no_prompts_removes_content_and_tool_call_arguments(self):
        logger = _logger()
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]
        event = MessageEvent(content="some text", role="assistant", tool_calls=tool_calls)
        with patch.dict(os.environ, {TRACELOOP_TRACE_CONTENT: "False"}):
            _emit_message_event(event, logger)
        body = _emitted_body(logger)
        assert "content" not in body
        assert "arguments" not in body["tool_calls"][0]["function"]


# ---------------------------------------------------------------------------
# _emit_choice_event
# ---------------------------------------------------------------------------


class TestEmitChoiceEvent:
    def test_non_assistant_role_keeps_role_in_message(self):
        logger = _logger()
        event = ChoiceEvent(
            index=0,
            message={"role": "system", "content": "oops"},
            finish_reason="stop",
        )
        _emit_choice_event(event, logger)
        body = _emitted_body(logger)
        assert body["message"].get("role") == "system"

    def test_not_none_tool_calls_kept_in_body(self):
        logger = _logger()
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        event = ChoiceEvent(
            index=0,
            message={"role": "assistant", "content": None},
            finish_reason="tool_call",
            tool_calls=tool_calls,
        )
        _emit_choice_event(event, logger)
        body = _emitted_body(logger)
        assert "tool_calls" in body

    def test_no_prompts_removes_content_and_role_from_message(self):
        logger = _logger()
        event = ChoiceEvent(
            index=0,
            message={"role": "assistant", "content": "hello"},
            finish_reason="stop",
        )
        with patch.dict(os.environ, {TRACELOOP_TRACE_CONTENT: "False"}):
            _emit_choice_event(event, logger)
        body = _emitted_body(logger)
        assert "content" not in body["message"]
        assert "role" not in body["message"]

    def test_no_prompts_removes_tool_call_arguments(self):
        logger = _logger()
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]
        event = ChoiceEvent(
            index=0,
            message={"role": "assistant", "content": None},
            finish_reason="tool_call",
            tool_calls=tool_calls,
        )
        with patch.dict(os.environ, {TRACELOOP_TRACE_CONTENT: "False"}):
            _emit_choice_event(event, logger)
        body = _emitted_body(logger)
        assert "arguments" not in body["tool_calls"][0]["function"]
