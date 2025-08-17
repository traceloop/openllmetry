"""
Test for the fix of the issue where assistant message content is missing
when tool calls are present in LangGraph/LangChain instrumentation.

This test reproduces the issue reported in GitHub where gen_ai.prompt.X.content
attributes were missing for assistant messages that contained tool_calls.
"""

import pytest
from unittest.mock import Mock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from opentelemetry.instrumentation.langchain.span_utils import set_chat_request
from opentelemetry.semconv_ai import SpanAttributes


def test_assistant_message_with_tool_calls_includes_content():
    """
    Test that when an assistant message has both content and tool_calls,
    both the content and tool_calls are included in the span attributes.
    
    This addresses the issue where content was missing when tool_calls were present.
    """
    # Create a mock span
    mock_span = Mock()
    mock_span.set_attribute = Mock()
    
    # Create a mock span_holder
    mock_span_holder = Mock()
    mock_span_holder.request_model = None
    
    # Create messages that reproduce the issue:
    # 1. User message
    # 2. Assistant message with BOTH content AND tool_calls
    messages = [[
        HumanMessage(content="what is the current time? First greet me."),
        AIMessage(
            content="Hello! Let me check the current time for you.",
            tool_calls=[{
                'id': 'call_qU7pH3EdQvzwkPyKPOdpgaKA',
                'name': 'get_current_time', 
                'args': {}
            }]
        ),
        ToolMessage(
            content="2025-08-15 08:15:21",
            tool_call_id="call_qU7pH3EdQvzwkPyKPOdpgaKA"
        ),
        AIMessage(content="The current time is 2025-08-15 08:15:21")
    ]]
    
    # Call the function that was previously buggy
    set_chat_request(mock_span, {}, messages, {}, mock_span_holder)
    
    # Verify that set_attribute was called with the expected attributes
    call_args = [call[0] for call in mock_span.set_attribute.call_args_list]
    
    # Extract all attribute names and values
    attributes = {args[0]: args[1] for args in call_args}
    
    # Check user message (prompt.0)
    assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "what is the current time? First greet me."
    
    # Check assistant message with tool calls (prompt.1)
    # This is the critical test - BOTH content AND tool_calls should be present
    assert f"{SpanAttributes.LLM_PROMPTS}.1.role" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
    
    # The fix should ensure that content is present even when tool_calls exist
    assert f"{SpanAttributes.LLM_PROMPTS}.1.content" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == "Hello! Let me check the current time for you."
    
    # Tool calls should also be present
    assert f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.id" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.id"] == "call_qU7pH3EdQvzwkPyKPOdpgaKA"
    assert f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.name" in attributes 
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.name"] == "get_current_time"
    
    # Check tool message (prompt.2)
    assert f"{SpanAttributes.LLM_PROMPTS}.2.role" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "tool"
    assert f"{SpanAttributes.LLM_PROMPTS}.2.content" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"] == "2025-08-15 08:15:21"
    assert f"{SpanAttributes.LLM_PROMPTS}.2.tool_call_id" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.2.tool_call_id"] == "call_qU7pH3EdQvzwkPyKPOdpgaKA"
    
    # Check final assistant message (prompt.3)
    assert f"{SpanAttributes.LLM_PROMPTS}.3.role" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.3.role"] == "assistant"
    assert f"{SpanAttributes.LLM_PROMPTS}.3.content" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.3.content"] == "The current time is 2025-08-15 08:15:21"


def test_assistant_message_with_only_tool_calls_no_content():
    """
    Test that when an assistant message has only tool_calls and no content,
    the tool_calls are still included and no content attribute is set.
    """
    # Create a mock span
    mock_span = Mock()
    mock_span.set_attribute = Mock()
    
    # Create a mock span_holder
    mock_span_holder = Mock()
    mock_span_holder.request_model = None
    
    # Create message with only tool_calls, no content
    messages = [[
        AIMessage(
            content="",  # Empty content
            tool_calls=[{
                'id': 'call_123',
                'name': 'some_tool', 
                'args': {'param': 'value'}
            }]
        )
    ]]
    
    # Call the function
    set_chat_request(mock_span, {}, messages, {}, mock_span_holder)
    
    # Verify that set_attribute was called with the expected attributes
    call_args = [call[0] for call in mock_span.set_attribute.call_args_list]
    
    # Extract all attribute names and values
    attributes = {args[0]: args[1] for args in call_args}
    
    # Check assistant message
    assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "assistant"
    
    # Content should NOT be set when it's empty (due to _set_span_attribute logic)
    # This is expected behavior to avoid cluttering spans with empty values
    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in attributes
    
    # Tool calls should be present
    assert f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.id" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.id"] == "call_123"
    assert f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.name" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.name"] == "some_tool"


def test_assistant_message_with_only_content_no_tool_calls():
    """
    Test that when an assistant message has only content and no tool_calls,
    the content is included and no tool_calls attributes are set.
    """
    # Create a mock span
    mock_span = Mock()
    mock_span.set_attribute = Mock()
    
    # Create a mock span_holder
    mock_span_holder = Mock()
    mock_span_holder.request_model = None
    
    # Create message with only content, no tool_calls
    messages = [[
        AIMessage(content="Just a regular response with no tool calls")
    ]]
    
    # Call the function
    set_chat_request(mock_span, {}, messages, {}, mock_span_holder)
    
    # Verify that set_attribute was called with the expected attributes
    call_args = [call[0] for call in mock_span.set_attribute.call_args_list]
    
    # Extract all attribute names and values
    attributes = {args[0]: args[1] for args in call_args}
    
    # Check assistant message
    assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "assistant"
    
    # Content should be present
    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" in attributes
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Just a regular response with no tool calls"
    
    # No tool call attributes should be present
    tool_call_attributes = [attr for attr in attributes.keys() if "tool_calls" in attr]
    assert len(tool_call_attributes) == 0