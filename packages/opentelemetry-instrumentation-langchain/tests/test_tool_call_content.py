"""
Test for the fix of the issue where assistant message content is missing
when tool calls are present in LangGraph/LangChain instrumentation.

This test reproduces the issue reported in GitHub where gen_ai.input_messages
attributes were missing content for assistant messages that contained tool_calls.
"""

import json
from unittest.mock import Mock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from opentelemetry.instrumentation.langchain.span_utils import set_chat_request
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def test_assistant_message_with_tool_calls_includes_content():
    """
    Test that when an assistant message has both content and tool_calls,
    both the content and tool_calls are included in the span attributes.

    This addresses the issue where content was missing when tool_calls were present.
    """
    mock_span = Mock()
    mock_span.set_attribute = Mock()
    mock_span_holder = Mock()
    mock_span_holder.request_model = None
    messages = [
        [
            HumanMessage(content="what is the current time? First greet me."),
            AIMessage(
                content="Hello! Let me check the current time for you.",
                tool_calls=[
                    {
                        "id": "call_qU7pH3EdQvzwkPyKPOdpgaKA",
                        "name": "get_current_time",
                        "args": {},
                    }
                ],
            ),
            ToolMessage(
                content="2025-08-15 08:15:21",
                tool_call_id="call_qU7pH3EdQvzwkPyKPOdpgaKA",
            ),
            AIMessage(content="The current time is 2025-08-15 08:15:21"),
        ]
    ]

    set_chat_request(mock_span, {}, messages, {}, mock_span_holder)

    call_args = [call[0] for call in mock_span.set_attribute.call_args_list]
    attributes = {args[0]: args[1] for args in call_args}

    input_messages = json.loads(attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])

    # Message 0: user message
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["type"] == "text"
    assert (
        input_messages[0]["parts"][0]["content"]
        == "what is the current time? First greet me."
    )

    # Message 1: assistant message with content and tool_calls
    assert input_messages[1]["role"] == "assistant"
    assert input_messages[1]["parts"][0]["type"] == "text"
    assert (
        input_messages[1]["parts"][0]["content"]
        == "Hello! Let me check the current time for you."
    )
    assert input_messages[1]["parts"][1]["type"] == "tool_call"
    assert input_messages[1]["parts"][1]["id"] == "call_qU7pH3EdQvzwkPyKPOdpgaKA"
    assert input_messages[1]["parts"][1]["name"] == "get_current_time"

    # Message 2: tool response message
    assert input_messages[2]["role"] == "tool"
    assert input_messages[2]["parts"][0]["type"] == "tool_call_response"
    assert input_messages[2]["parts"][0]["id"] == "call_qU7pH3EdQvzwkPyKPOdpgaKA"
    assert input_messages[2]["parts"][0]["response"] == "2025-08-15 08:15:21"

    # Message 3: assistant message with only content
    assert input_messages[3]["role"] == "assistant"
    assert input_messages[3]["parts"][0]["type"] == "text"
    assert (
        input_messages[3]["parts"][0]["content"]
        == "The current time is 2025-08-15 08:15:21"
    )


def test_assistant_message_with_only_tool_calls_no_content():
    """
    Test that when an assistant message has only tool_calls and no content,
    the tool_calls are still included and no text part is set.
    """
    mock_span = Mock()
    mock_span.set_attribute = Mock()
    mock_span_holder = Mock()
    mock_span_holder.request_model = None

    messages = [
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_123", "name": "some_tool", "args": {"param": "value"}}
                ],
            )
        ]
    ]

    set_chat_request(mock_span, {}, messages, {}, mock_span_holder)

    call_args = [call[0] for call in mock_span.set_attribute.call_args_list]
    attributes = {args[0]: args[1] for args in call_args}

    input_messages = json.loads(attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])

    assert input_messages[0]["role"] == "assistant"
    # No text part since content is empty; only tool_call parts
    tool_call_parts = [p for p in input_messages[0]["parts"] if p["type"] == "tool_call"]
    assert len(tool_call_parts) == 1
    assert tool_call_parts[0]["id"] == "call_123"
    assert tool_call_parts[0]["name"] == "some_tool"

    text_parts = [p for p in input_messages[0]["parts"] if p["type"] == "text"]
    assert len(text_parts) == 0


def test_assistant_message_with_only_content_no_tool_calls():
    """
    Test that when an assistant message has only content and no tool_calls,
    the content is included and no tool_call parts are set.
    """
    mock_span = Mock()
    mock_span.set_attribute = Mock()
    mock_span_holder = Mock()
    mock_span_holder.request_model = None

    messages = [[AIMessage(content="Just a regular response with no tool calls")]]

    set_chat_request(mock_span, {}, messages, {}, mock_span_holder)

    call_args = [call[0] for call in mock_span.set_attribute.call_args_list]
    attributes = {args[0]: args[1] for args in call_args}

    input_messages = json.loads(attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])

    assert input_messages[0]["role"] == "assistant"
    assert input_messages[0]["parts"][0]["type"] == "text"
    assert (
        input_messages[0]["parts"][0]["content"]
        == "Just a regular response with no tool calls"
    )

    tool_call_parts = [p for p in input_messages[0]["parts"] if p["type"] == "tool_call"]
    assert len(tool_call_parts) == 0
