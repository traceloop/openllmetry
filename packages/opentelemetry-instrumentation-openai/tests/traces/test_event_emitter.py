"""Unit tests for event_emitter behavior with optional tool call arguments.

These tests verify that emitting events with tool calls that omit the
`arguments` key does not raise errors, and that the _FunctionToolCall
TypedDict correctly allows arguments to be absent.
"""

from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.instrumentation.openai.shared.event_models import (
    ChoiceEvent,
    MessageEvent,
    ToolCall,
    _FunctionToolCall,
)
from opentelemetry.instrumentation.openai.shared.event_emitter import (
    emit_event,
)


@pytest.fixture
def mock_event_logger():
    logger = MagicMock()
    with patch(
        "opentelemetry.instrumentation.openai.shared.event_emitter.Config"
    ) as mock_config:
        mock_config.event_logger = logger
        yield logger


def _make_tool_call_without_arguments() -> ToolCall:
    """Build a ToolCall whose _FunctionToolCall omits the arguments key."""
    function: _FunctionToolCall = {"name": "get_weather"}  # no arguments key
    return {"id": "call_123", "type": "function", "function": function}


def _make_tool_call_with_none_arguments() -> ToolCall:
    function: _FunctionToolCall = {"name": "get_weather", "arguments": None}
    return {"id": "call_456", "type": "function", "function": function}


def _make_tool_call_with_arguments() -> ToolCall:
    function: _FunctionToolCall = {
        "name": "get_weather",
        "arguments": '{"location": "NYC"}',
    }
    return {"id": "call_789", "type": "function", "function": function}


class TestFunctionToolCallTypedDict:
    def test_can_instantiate_without_arguments(self):
        """_FunctionToolCall should be constructable without the arguments key."""
        func: _FunctionToolCall = {"name": "my_tool"}
        assert func["name"] == "my_tool"
        assert "arguments" not in func

    def test_can_instantiate_with_none_arguments(self):
        func: _FunctionToolCall = {"name": "my_tool", "arguments": None}
        assert func["arguments"] is None

    def test_can_instantiate_with_string_arguments(self):
        func: _FunctionToolCall = {"name": "my_tool", "arguments": "{}"}
        assert func["arguments"] == "{}"


class TestEmitMessageEventWithMissingArguments:
    def test_emit_with_send_prompts_arguments_absent(self, mock_event_logger):
        """Emitting a message event with arguments absent should not raise."""
        event = MessageEvent(
            role="assistant",
            content=None,
            tool_calls=[_make_tool_call_without_arguments()],
        )
        with (
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_emit_events",
                return_value=True,
            ),
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_send_prompts",
                return_value=True,
            ),
        ):
            emit_event(event)

        mock_event_logger.emit.assert_called_once()
        body = mock_event_logger.emit.call_args[0][0].body
        assert body["tool_calls"][0]["function"]["name"] == "get_weather"
        assert "arguments" not in body["tool_calls"][0]["function"]

    def test_emit_with_no_send_prompts_arguments_absent(self, mock_event_logger):
        """When should_send_prompts=False, pop on absent arguments key should not raise."""
        event = MessageEvent(
            role="assistant",
            content=None,
            tool_calls=[_make_tool_call_without_arguments()],
        )
        with (
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_emit_events",
                return_value=True,
            ),
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_send_prompts",
                return_value=False,
            ),
        ):
            emit_event(event)

        mock_event_logger.emit.assert_called_once()
        body = mock_event_logger.emit.call_args[0][0].body
        assert "arguments" not in body["tool_calls"][0]["function"]

    def test_emit_with_no_send_prompts_arguments_none(self, mock_event_logger):
        """When should_send_prompts=False, arguments=None is popped without error."""
        event = MessageEvent(
            role="assistant",
            content=None,
            tool_calls=[_make_tool_call_with_none_arguments()],
        )
        with (
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_emit_events",
                return_value=True,
            ),
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_send_prompts",
                return_value=False,
            ),
        ):
            emit_event(event)

        mock_event_logger.emit.assert_called_once()
        body = mock_event_logger.emit.call_args[0][0].body
        assert "arguments" not in body["tool_calls"][0]["function"]

    def test_emit_with_no_send_prompts_arguments_present(self, mock_event_logger):
        """When should_send_prompts=False, present arguments are stripped."""
        event = MessageEvent(
            role="assistant",
            content=None,
            tool_calls=[_make_tool_call_with_arguments()],
        )
        with (
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_emit_events",
                return_value=True,
            ),
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_send_prompts",
                return_value=False,
            ),
        ):
            emit_event(event)

        mock_event_logger.emit.assert_called_once()
        body = mock_event_logger.emit.call_args[0][0].body
        assert "arguments" not in body["tool_calls"][0]["function"]


class TestEmitChoiceEventWithMissingArguments:
    def test_emit_with_send_prompts_arguments_absent(self, mock_event_logger):
        """Emitting a choice event with arguments absent should not raise."""
        event = ChoiceEvent(
            index=0,
            message={"content": None, "role": "assistant"},
            finish_reason="tool_calls",
            tool_calls=[_make_tool_call_without_arguments()],
        )
        with (
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_emit_events",
                return_value=True,
            ),
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_send_prompts",
                return_value=True,
            ),
        ):
            emit_event(event)

        mock_event_logger.emit.assert_called_once()
        body = mock_event_logger.emit.call_args[0][0].body
        assert body["tool_calls"][0]["function"]["name"] == "get_weather"
        assert "arguments" not in body["tool_calls"][0]["function"]

    def test_emit_with_no_send_prompts_arguments_absent(self, mock_event_logger):
        """When should_send_prompts=False, pop on absent arguments should not raise."""
        event = ChoiceEvent(
            index=0,
            message={"content": None, "role": "assistant"},
            finish_reason="tool_calls",
            tool_calls=[_make_tool_call_without_arguments()],
        )
        with (
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_emit_events",
                return_value=True,
            ),
            patch(
                "opentelemetry.instrumentation.openai.shared.event_emitter.should_send_prompts",
                return_value=False,
            ),
        ):
            emit_event(event)

        mock_event_logger.emit.assert_called_once()
        body = mock_event_logger.emit.call_args[0][0].body
        assert "arguments" not in body["tool_calls"][0]["function"]
