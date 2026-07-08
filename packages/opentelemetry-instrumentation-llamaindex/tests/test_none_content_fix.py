"""Test for None content handling in span_utils - Issue #3513"""

import json

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
        Test that output message has empty parts when content is None (StructuredLLM case).
        """
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_message = MagicMock()
        mock_message.role = MessageRole.ASSISTANT
        mock_message.content = None
        mock_message.additional_kwargs = {}

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.raw = {}

        mock_event = MagicMock()
        mock_event.response = mock_response
        mock_event.messages = []

        with patch('opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts', return_value=True):
            set_llm_chat_response(mock_event, mock_span)

        # Find the gen_ai.output.messages call and verify empty parts for None content
        raw = None
        for call in mock_span.set_attribute.call_args_list:
            if call.args[0] == GenAIAttributes.GEN_AI_OUTPUT_MESSAGES:
                raw = call.args[1]
        assert raw is not None
        msgs = json.loads(raw)
        assert msgs[0]["parts"] == [], "Parts should be empty when content is None"

    def test_set_llm_chat_response_with_valid_content(self):
        """
        Test that set_llm_chat_response correctly sets content as JSON when it's not None.
        """
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_message = MagicMock()
        mock_message.role = MessageRole.ASSISTANT
        mock_message.content = "This is a valid response"
        mock_message.additional_kwargs = {}

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.raw = {}

        mock_event = MagicMock()
        mock_event.response = mock_response
        mock_event.messages = []

        with patch('opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts', return_value=True):
            set_llm_chat_response(mock_event, mock_span)

        # Verify content was set in JSON output messages
        raw = None
        for call in mock_span.set_attribute.call_args_list:
            if call.args[0] == GenAIAttributes.GEN_AI_OUTPUT_MESSAGES:
                raw = call.args[1]
        assert raw is not None
        msgs = json.loads(raw)
        assert msgs[0]["parts"][0]["content"] == "This is a valid response"

    def test_set_llm_predict_response_with_none_output(self):
        """
        Test that predict response handles None output gracefully (empty text).
        """
        mock_span = MagicMock()

        mock_event = MagicMock()
        mock_event.output = None

        with patch('opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts', return_value=True):
            set_llm_predict_response(mock_event, mock_span)

        raw = None
        for call in mock_span.set_attribute.call_args_list:
            if call.args[0] == GenAIAttributes.GEN_AI_OUTPUT_MESSAGES:
                raw = call.args[1]
        assert raw is not None
        msgs = json.loads(raw)
        assert msgs[0]["role"] == "assistant"
        # None output → empty string → empty parts (build_completion_output_message)
        # Actually: event.output or "" → "", which gives empty parts
        assert msgs[0]["parts"] == []

    def test_set_llm_predict_response_with_valid_output(self):
        """
        Test that set_llm_predict_response correctly sets content as JSON.
        """
        mock_span = MagicMock()

        mock_event = MagicMock()
        mock_event.output = "Valid output text"

        with patch('opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts', return_value=True):
            set_llm_predict_response(mock_event, mock_span)

        raw = None
        for call in mock_span.set_attribute.call_args_list:
            if call.args[0] == GenAIAttributes.GEN_AI_OUTPUT_MESSAGES:
                raw = call.args[1]
        assert raw is not None
        msgs = json.loads(raw)
        assert msgs[0]["parts"][0]["content"] == "Valid output text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
