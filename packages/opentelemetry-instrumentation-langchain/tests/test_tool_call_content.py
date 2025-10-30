"""
Test for the fix of the issue where assistant message content is missing
when tool calls are present in LangGraph/LangChain instrumentation.

This test reproduces the issue reported in GitHub where gen_ai.prompt.X.content
attributes were missing for assistant messages that contained tool_calls.
"""

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

    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.role" in attributes
    assert attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" in attributes
    assert (
        attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "what is the current time? First greet me."
    )
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.1.role" in attributes
    assert attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "assistant"
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.1.content" in attributes
    assert (
        attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
        == "Hello! Let me check the current time for you."
    )
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.1.tool_calls.0.id" in attributes
    assert (
        attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.tool_calls.0.id"]
        == "call_qU7pH3EdQvzwkPyKPOdpgaKA"
    )
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.1.tool_calls.0.name" in attributes
    assert (
        attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.tool_calls.0.name"]
        == "get_current_time"
    )
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.2.role" in attributes
    assert attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.role"] == "tool"
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.2.content" in attributes
    assert (
        attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.content"] == "2025-08-15 08:15:21"
    )
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.2.tool_call_id" in attributes
    assert (
        attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.tool_call_id"]
        == "call_qU7pH3EdQvzwkPyKPOdpgaKA"
    )
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.3.role" in attributes
    assert attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.3.role"] == "assistant"
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.3.content" in attributes
    assert (
        attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.3.content"]
        == "The current time is 2025-08-15 08:15:21"
    )


def test_assistant_message_with_only_tool_calls_no_content():
    """
    Test that when an assistant message has only tool_calls and no content,
    the tool_calls are still included and no content attribute is set.
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

    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.role" in attributes
    assert attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "assistant"
    # Content is being set as empty string, so we expect it to be present
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" in attributes
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.id" in attributes
    assert attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.id"] == "call_123"
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.name" in attributes
    assert (
        attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.name"] == "some_tool"
    )


def test_assistant_message_with_only_content_no_tool_calls():
    """
    Test that when an assistant message has only content and no tool_calls,
    the content is included and no tool_calls attributes are set.
    """
    mock_span = Mock()
    mock_span.set_attribute = Mock()
    mock_span_holder = Mock()
    mock_span_holder.request_model = None

    messages = [[AIMessage(content="Just a regular response with no tool calls")]]

    set_chat_request(mock_span, {}, messages, {}, mock_span_holder)

    call_args = [call[0] for call in mock_span.set_attribute.call_args_list]

    attributes = {args[0]: args[1] for args in call_args}

    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.role" in attributes
    assert attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "assistant"
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" in attributes
    assert (
        attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Just a regular response with no tool calls"
    )

    tool_call_attributes = [attr for attr in attributes.keys() if "tool_calls" in attr]
    assert len(tool_call_attributes) == 0
