"""
Tests for generation role extraction in completion spans.

This tests the fix for generation.type returning class names ("ChatGeneration", "Generation")
instead of message types ("ai", "tool", etc.), which caused completion roles to appear as "unknown"
in observability traces.
"""

import pytest
from unittest.mock import Mock
from langchain_core.outputs import LLMResult, ChatGeneration, Generation
from langchain_core.messages import AIMessage, ToolMessage
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.instrumentation.langchain.span_utils import set_chat_response


class TestCompletionRoleExtraction:
    """Test that completion roles are correctly extracted from generation objects."""

    @pytest.fixture
    def mock_span(self):
        """Create a mock span for testing."""
        span = Mock()
        span.is_recording.return_value = True
        span.attributes = {}

        def set_attribute(key, value):
            span.attributes[key] = value

        span.set_attribute = set_attribute
        return span

    def test_chat_generation_with_ai_message_role(self, mock_span, monkeypatch):
        """Test that ChatGeneration with AIMessage correctly extracts 'assistant' role."""
        # Mock should_send_prompts to return True
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        # Create ChatGeneration with AIMessage
        generation = ChatGeneration(message=AIMessage(content="Hello!"))
        llm_result = LLMResult(generations=[[generation]])

        # Call the function
        set_chat_response(mock_span, llm_result)

        # Assert role is 'assistant', not 'unknown'
        role_key = f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"
        assert role_key in mock_span.attributes
        assert mock_span.attributes[role_key] == "assistant"

    def test_chat_generation_with_tool_message_role(self, mock_span, monkeypatch):
        """Test that ChatGeneration with ToolMessage correctly extracts 'tool' role."""
        # Mock should_send_prompts to return True
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        # Create ChatGeneration with ToolMessage
        generation = ChatGeneration(
            message=ToolMessage(content="Tool result", tool_call_id="123")
        )
        llm_result = LLMResult(generations=[[generation]])

        # Call the function
        set_chat_response(mock_span, llm_result)

        # Assert role is 'tool', not 'unknown'
        role_key = f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"
        assert role_key in mock_span.attributes
        assert mock_span.attributes[role_key] == "tool"

    def test_generation_without_message_defaults_to_assistant(self, mock_span, monkeypatch):
        """Test that Generation (non-chat) defaults to 'assistant' role."""
        # Mock should_send_prompts to return True
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        # Create Generation without message (legacy completion)
        generation = Generation(text="This is a completion")
        llm_result = LLMResult(generations=[[generation]])

        # Call the function
        set_chat_response(mock_span, llm_result)

        # Assert role defaults to 'assistant', not 'unknown'
        role_key = f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"
        assert role_key in mock_span.attributes
        assert mock_span.attributes[role_key] == "assistant"

    def test_multiple_generations_with_different_roles(self, mock_span, monkeypatch):
        """Test that multiple generations with different message types are handled correctly."""
        # Mock should_send_prompts to return True
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        # Create multiple generations with different message types
        gen1 = ChatGeneration(message=AIMessage(content="AI response"))
        gen2 = ChatGeneration(message=ToolMessage(content="Tool result", tool_call_id="123"))
        gen3 = Generation(text="Legacy completion")

        llm_result = LLMResult(generations=[[gen1], [gen2], [gen3]])

        # Call the function
        set_chat_response(mock_span, llm_result)

        # Assert all roles are correctly set
        assert mock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"] == "assistant"
        assert mock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.1.role"] == "tool"
        assert mock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.2.role"] == "assistant"

    def test_generation_type_attribute_is_not_used(self, mock_span, monkeypatch):
        """Test that generation.type (which returns class name) is not used directly."""
        # Mock should_send_prompts to return True
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        # Create ChatGeneration - note that generation.type would be "ChatGeneration"
        generation = ChatGeneration(message=AIMessage(content="Test"))

        # Verify the bug scenario: generation.type returns class name, not message type
        assert generation.type == "ChatGeneration"  # This is the bug
        assert generation.message.type == "ai"  # This is what we should use

        llm_result = LLMResult(generations=[[generation]])

        # Call the function
        set_chat_response(mock_span, llm_result)

        # Assert role is 'assistant', not 'unknown'
        # If the bug existed, passing generation.type directly to _message_type_to_role
        # would return 'unknown' because "ChatGeneration" doesn't match any message type
        role_key = f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"
        assert mock_span.attributes[role_key] == "assistant"
