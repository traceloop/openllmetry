"""Test for None content handling in span_utils - Issue #3513"""

import pytest
from unittest.mock import MagicMock, patch
from llama_index.core.base.llms.types import MessageRole

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

# Import the functions we're testing
from opentelemetry.instrumentation.llamaindex.span_utils import (
    _set_span_attribute,
    set_llm_chat_response,
    set_llm_predict_response,
)


class TestSetSpanAttribute:
    """Tests for the _set_span_attribute utility function."""

    def test_set_span_attribute_with_valid_value(self):
        """Test that _set_span_attribute sets the attribute when value is valid."""
        mock_span = MagicMock()
        _set_span_attribute(mock_span, "test.attribute", "valid_value")
        mock_span.set_attribute.assert_called_once_with("test.attribute", "valid_value")

    def test_set_span_attribute_with_none_value(self):
        """Test that _set_span_attribute does NOT set the attribute when value is None."""
        mock_span = MagicMock()
        _set_span_attribute(mock_span, "test.attribute", None)
        mock_span.set_attribute.assert_not_called()

    def test_set_span_attribute_with_empty_string(self):
        """Test that _set_span_attribute does NOT set the attribute when value is empty string."""
        mock_span = MagicMock()
        _set_span_attribute(mock_span, "test.attribute", "")
        mock_span.set_attribute.assert_not_called()


class TestNoneContentHandling:
    """Tests for handling None content in StructuredLLM responses."""

    def test_set_llm_chat_response_with_none_content(self):
        """
        Test that set_llm_chat_response doesn't set gen_ai.completion.0.content
        when response.message.content is None (StructuredLLM case).
        """
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        mock_message = MagicMock()
        mock_message.role = MessageRole.ASSISTANT
        mock_message.content = None
        
        mock_response = MagicMock()
        mock_response.message = mock_message
        
        mock_event = MagicMock()
        mock_event.response = mock_response
        mock_event.messages = []
        
        with patch('opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts', return_value=True):
            set_llm_chat_response(mock_event, mock_span)
        
        # Verify role was set
        mock_span.set_attribute.assert_any_call(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role",
            MessageRole.ASSISTANT.value
        )
        
        # Verify content was NOT set (no call with the content attribute key)
        content_key = f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
        assert not any(
            c.args[0] == content_key for c in mock_span.set_attribute.call_args_list
        ), "Content attribute should NOT be set when value is None"

    def test_set_llm_chat_response_with_valid_content(self):
        """
        Test that set_llm_chat_response correctly sets content when it's not None.
        """
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        mock_message = MagicMock()
        mock_message.role = MessageRole.ASSISTANT
        mock_message.content = "This is a valid response"
        
        mock_response = MagicMock()
        mock_response.message = mock_message
        
        mock_event = MagicMock()
        mock_event.response = mock_response
        mock_event.messages = []
        
        with patch('opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts', return_value=True):
            set_llm_chat_response(mock_event, mock_span)
        
        # Verify content was set
        mock_span.set_attribute.assert_any_call(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content",
            "This is a valid response"
        )

    def test_set_llm_predict_response_with_none_output(self):
        """
        Test that set_llm_predict_response doesn't set gen_ai.completion.content
        when event.output is None.
        """
        mock_span = MagicMock()
        
        mock_event = MagicMock()
        mock_event.output = None
        
        with patch('opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts', return_value=True):
            set_llm_predict_response(mock_event, mock_span)
        
        # Verify role was set
        mock_span.set_attribute.assert_any_call(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.role",
            MessageRole.ASSISTANT.value
        )
        
        # Verify content was NOT set
        content_key = f"{GenAIAttributes.GEN_AI_COMPLETION}.content"
        assert not any(
            c.args[0] == content_key for c in mock_span.set_attribute.call_args_list
        ), "Content attribute should NOT be set when value is None"

    def test_set_llm_predict_response_with_valid_output(self):
        """
        Test that set_llm_predict_response correctly sets content when output is not None.
        """
        mock_span = MagicMock()
        
        mock_event = MagicMock()
        mock_event.output = "Valid output text"
        
        with patch('opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts', return_value=True):
            set_llm_predict_response(mock_event, mock_span)
        
        # Verify content was set
        mock_span.set_attribute.assert_any_call(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.content",
            "Valid output text"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
