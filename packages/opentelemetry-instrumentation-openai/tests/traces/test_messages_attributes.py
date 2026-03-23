"""Unit tests for gen_ai.input.messages / gen_ai.output.messages formatting.

These test the _set_input_messages and _set_output_messages code paths
using mocked spans, without VCR cassettes.
"""

import json
from unittest.mock import MagicMock

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _set_input_messages,
    _set_output_messages,
)


@pytest.fixture
def mock_span():
    span = MagicMock()
    span.is_recording.return_value = True
    attrs = {}

    def set_attribute(name, value):
        attrs[name] = value

    span.set_attribute = set_attribute
    span._attrs = attrs
    return span


def _get_input_messages(span):
    return json.loads(span._attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])


def _get_output_messages(span):
    return json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])


# ---------------------------------------------------------------------------
# Multi-turn input messages (system + user + assistant with tool_calls + tool)
# ---------------------------------------------------------------------------

class TestSetInputMessages:
    def test_multiturn_with_tool_calls(self, mock_span):
        messages = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "What is the weather in NYC?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": '{"temp": 72, "unit": "F"}',
            },
        ]

        _set_input_messages(mock_span, messages)
        result = _get_input_messages(mock_span)

        assert len(result) == 4

        # system
        assert result[0]["role"] == "system"
        assert result[0]["parts"][0]["type"] == "text"
        assert result[0]["parts"][0]["content"] == "You are a weather assistant."

        # user
        assert result[1]["role"] == "user"
        assert result[1]["parts"][0]["content"] == "What is the weather in NYC?"

        # assistant with tool call (content=None → empty parts + tool_call_request)
        assert result[2]["role"] == "assistant"
        assert any(p["type"] == "tool_call" for p in result[2]["parts"])
        tool_part = next(
            p for p in result[2]["parts"] if p["type"] == "tool_call"
        )
        assert tool_part["name"] == "get_weather"
        assert tool_part["id"] == "call_abc123"
        assert tool_part["arguments"] == '{"location": "NYC"}'

        # tool response
        assert result[3]["role"] == "tool"
        assert result[3]["parts"][0]["type"] == "tool_call_response"
        assert result[3]["parts"][0]["id"] == "call_abc123"
        assert result[3]["parts"][0]["response"] == '{"temp": 72, "unit": "F"}'

    def test_user_with_none_content(self, mock_span):
        """User message with None content should not crash or produce null parts."""
        _set_input_messages(mock_span, [{"role": "user", "content": None}])
        result = _get_input_messages(mock_span)

        assert len(result) == 1
        assert result[0]["parts"] is not None

    def test_system_with_none_content(self, mock_span):
        """System message with None content should not crash or produce null parts."""
        _set_input_messages(mock_span, [{"role": "system", "content": None}])
        result = _get_input_messages(mock_span)

        assert len(result) == 1
        assert result[0]["parts"] is not None

    def test_developer_role(self, mock_span):
        _set_input_messages(mock_span, [{"role": "developer", "content": "Be concise."}])
        result = _get_input_messages(mock_span)

        assert result[0]["role"] == "developer"
        assert result[0]["parts"][0]["content"] == "Be concise."

    def test_none_messages(self, mock_span):
        """None messages input should be a no-op."""
        _set_input_messages(mock_span, None)
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in mock_span._attrs

    def test_assistant_with_text_and_tool_calls(self, mock_span):
        """Assistant message with both text content and tool calls."""
        _set_input_messages(mock_span, [
            {
                "role": "assistant",
                "content": "Let me check that for you.",
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        ])
        result = _get_input_messages(mock_span)
        parts = result[0]["parts"]

        text_parts = [p for p in parts if p.get("type") == "text"]
        tool_parts = [p for p in parts if p.get("type") == "tool_call"]
        assert len(text_parts) == 1
        assert text_parts[0]["content"] == "Let me check that for you."
        assert len(tool_parts) == 1
        assert tool_parts[0]["name"] == "lookup"


# ---------------------------------------------------------------------------
# Output messages
# ---------------------------------------------------------------------------

class TestSetOutputMessages:
    def test_text_response(self, mock_span):
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["finish_reason"] == "stop"
        assert result[0]["parts"][0]["content"] == "Hello!"
        assert result[0]["parts"][0]["type"] == "text"

    def test_tool_call_response(self, mock_span):
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)

        assert len(result) == 1
        msg = result[0]
        assert msg["finish_reason"] == "tool_call"
        tool_parts = [p for p in msg["parts"] if p["type"] == "tool_call"]
        assert len(tool_parts) == 1
        assert tool_parts[0]["name"] == "get_weather"
        assert tool_parts[0]["id"] == "call_abc"

    def test_none_message_in_choice(self, mock_span):
        """Choice with message=None should not crash."""
        choices = [
            {"index": 0, "message": None, "finish_reason": "content_filter"},
        ]
        _set_output_messages(mock_span, choices)
        # Should not crash; attribute may or may not be set

    def test_none_choices(self, mock_span):
        """None choices should be a no-op."""
        _set_output_messages(mock_span, None)
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in mock_span._attrs

    def test_multiple_choices(self, mock_span):
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Answer A"},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "Answer B"},
                "finish_reason": "stop",
            },
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)

        assert len(result) == 2
        assert result[0]["parts"][0]["content"] == "Answer A"
        assert result[1]["parts"][0]["content"] == "Answer B"
