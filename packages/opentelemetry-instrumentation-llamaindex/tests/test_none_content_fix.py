"""Test for None content handling in span_utils - Issue #3513"""

import pytest
from unittest.mock import MagicMock, patch
from llama_index.core.base.llms.types import MessageRole, ChatMessage, ChatResponse

# Import the functions we're testing
from opentelemetry.instrumentation.llamaindex.span_utils import (
    set_llm_chat_response,
    set_llm_predict_response,
)


class TestNoneContentHandling:
    """Tests for handling None content in StructuredLLM responses."""

    def test_set_llm_chat_response_with_none_content(self):
        """
        Test that set_llm_chat_response doesn't set gen_ai.completion.0.content
        when response.message.content is None (StructuredLLM case).
        
        This reproduces issue #3513 where StructuredLLM returns None for
        response.message.content because the structured output goes to response.raw.
        """
        # Create a mock span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        # Create a mock response with None content (simulating StructuredLLM)
        mock_message = MagicMock()
        mock_message.role = MessageRole.ASSISTANT
        mock_message.content = None  # This is the key - StructuredLLM returns None here
        
        mock_response = MagicMock()
        mock_response.message = mock_message
        
        # Create a mock event
        mock_event = MagicMock()
        mock_event.response = mock_response
        mock_event.messages = []
        
        # Patch should_send_prompts to return True
        with patch('opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts', return_value=True):
            # Call the function - this should NOT raise an error or set None attribute
            set_llm_chat_response(mock_event, mock_span)
        
        # Verify that set_attribute was called for role
        role_calls = [call for call in mock_span.set_attribute.call_args_list 
                      if 'role' in str(call)]
        assert len(role_calls) > 0, "Role attribute should be set"
        
        # Verify that set_attribute was NOT called with None for content
        content_calls = [call for call in mock_span.set_attribute.call_args_list 
                         if 'content' in str(call) and call[0][1] is None]
        assert len(content_calls) == 0, "Content attribute should NOT be set to None"

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
        content_calls = [call for call in mock_span.set_attribute.call_args_list 
                         if 'completion' in str(call) and 'content' in str(call)]
        assert len(content_calls) > 0, "Content attribute should be set when not None"

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
        role_calls = [call for call in mock_span.set_attribute.call_args_list 
                      if 'role' in str(call)]
        assert len(role_calls) > 0, "Role attribute should be set"
        
        # Verify content was NOT set to None
        content_calls = [call for call in mock_span.set_attribute.call_args_list 
                         if 'content' in str(call) and call[0][1] is None]
        assert len(content_calls) == 0, "Content attribute should NOT be set to None"

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
        content_calls = [call for call in mock_span.set_attribute.call_args_list 
                         if 'content' in str(call)]
        assert len(content_calls) > 0, "Content attribute should be set when not None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
