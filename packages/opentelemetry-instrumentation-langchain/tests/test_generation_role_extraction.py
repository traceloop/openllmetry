"""
Tests for generation role extraction in completion spans.

This tests the fix for generation.type returning class names ("ChatGeneration", "Generation")
instead of message types ("ai", "tool", etc.), which caused completion roles to appear as "unknown"
in observability traces.
"""

import json
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
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        generation = ChatGeneration(message=AIMessage(content="Hello!"))
        llm_result = LLMResult(generations=[[generation]])

        set_chat_response(mock_span, llm_result)

        output_messages = json.loads(mock_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert output_messages[0]["role"] == "assistant"

    def test_chat_generation_with_tool_message_role(self, mock_span, monkeypatch):
        """Test that ChatGeneration with ToolMessage correctly extracts 'tool' role."""
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        generation = ChatGeneration(
            message=ToolMessage(content="Tool result", tool_call_id="123")
        )
        llm_result = LLMResult(generations=[[generation]])

        set_chat_response(mock_span, llm_result)

        output_messages = json.loads(mock_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert output_messages[0]["role"] == "tool"

    def test_generation_without_message_defaults_to_assistant(self, mock_span, monkeypatch):
        """Test that Generation (non-chat) defaults to 'assistant' role."""
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        generation = Generation(text="This is a completion")
        llm_result = LLMResult(generations=[[generation]])

        set_chat_response(mock_span, llm_result)

        output_messages = json.loads(mock_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert output_messages[0]["role"] == "assistant"

    def test_multiple_generations_with_different_roles(self, mock_span, monkeypatch):
        """Test that multiple generations with different message types are handled correctly."""
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        gen1 = ChatGeneration(message=AIMessage(content="AI response"))
        gen2 = ChatGeneration(message=ToolMessage(content="Tool result", tool_call_id="123"))
        gen3 = Generation(text="Legacy completion")

        llm_result = LLMResult(generations=[[gen1], [gen2], [gen3]])

        set_chat_response(mock_span, llm_result)

        output_messages = json.loads(mock_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert output_messages[0]["role"] == "assistant"
        assert output_messages[1]["role"] == "tool"
        assert output_messages[2]["role"] == "assistant"

    def test_generation_type_attribute_is_not_used(self, mock_span, monkeypatch):
        """Test that generation.type (which returns class name) is not used directly."""
        monkeypatch.setattr(
            "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
            lambda: True
        )

        generation = ChatGeneration(message=AIMessage(content="Test"))

        # Verify the bug scenario: generation.type returns class name, not message type
        assert generation.type == "ChatGeneration"  # This is the bug
        assert generation.message.type == "ai"  # This is what we should use

        llm_result = LLMResult(generations=[[generation]])

        set_chat_response(mock_span, llm_result)

        output_messages = json.loads(mock_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert output_messages[0]["role"] == "assistant"
