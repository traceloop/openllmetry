import json
import pytest
from unittest.mock import Mock, patch
from opentelemetry.instrumentation.vertexai.span_utils import (
    set_input_attributes,
    set_input_attributes_sync,
)
from opentelemetry.semconv_ai import SpanAttributes


class TestRoleAttributes:
    """Test cases for role attribute handling in VertexAI instrumentation"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_span = Mock()
        self.mock_span.is_recording.return_value = True
        self.mock_span.context.trace_id = "test_trace_id"
        self.mock_span.context.span_id = "test_span_id"
        self.span_attributes = {}

        # Mock set_attribute to capture the attributes
        def capture_attribute(key, value):
            self.span_attributes[key] = value

        self.mock_span.set_attribute = capture_attribute

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    @pytest.mark.asyncio
    async def test_async_role_attribute_for_string_args(self, mock_should_send_prompts):
        """Test that role='user' is set for string arguments in async function"""
        mock_should_send_prompts.return_value = True

        # Test with string arguments
        args = ["Hello, world!"]
        await set_input_attributes(self.mock_span, args)

        # Verify role attribute is set
        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

        # Verify content is also set
        assert f"{SpanAttributes.LLM_PROMPTS}.0.content" in self.span_attributes
        expected_content = json.dumps([{"type": "text", "text": "Hello, world!"}])
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == expected_content

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    @pytest.mark.asyncio
    async def test_async_role_attribute_for_multiple_args(self, mock_should_send_prompts):
        """Test that role='user' is set for multiple arguments in async function"""
        mock_should_send_prompts.return_value = True

        # Test with multiple string arguments
        args = ["First message", "Second message"]
        await set_input_attributes(self.mock_span, args)

        # Verify role attributes are set for both arguments
        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

        assert f"{SpanAttributes.LLM_PROMPTS}.1.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    @pytest.mark.asyncio
    async def test_async_role_attribute_with_mixed_content(self, mock_should_send_prompts):
        """Test role attribute with mixed content types (text + image placeholders)"""
        mock_should_send_prompts.return_value = True

        # Test with list containing mixed content
        args = [["Text content", "More text"]]
        await set_input_attributes(self.mock_span, args)

        # Verify role attribute is set
        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_role_attribute_for_string_args(self, mock_should_send_prompts):
        """Test that role='user' is set for string arguments in sync function"""
        mock_should_send_prompts.return_value = True

        # Test with string arguments
        args = ["Hello, world!"]
        set_input_attributes_sync(self.mock_span, args)

        # Verify role attribute is set
        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

        # Verify content is also set
        assert f"{SpanAttributes.LLM_PROMPTS}.0.content" in self.span_attributes
        expected_content = json.dumps([{"type": "text", "text": "Hello, world!"}])
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == expected_content

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_role_attribute_for_multiple_args(self, mock_should_send_prompts):
        """Test that role='user' is set for multiple arguments in sync function"""
        mock_should_send_prompts.return_value = True

        # Test with multiple string arguments
        args = ["First message", "Second message"]
        set_input_attributes_sync(self.mock_span, args)

        # Verify role attributes are set for both arguments
        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

        assert f"{SpanAttributes.LLM_PROMPTS}.1.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_role_attribute_with_mixed_content(self, mock_should_send_prompts):
        """Test role attribute with mixed content types in sync function"""
        mock_should_send_prompts.return_value = True

        # Test with list containing mixed content
        args = [["Text content", "More text"]]
        set_input_attributes_sync(self.mock_span, args)

        # Verify role attribute is set
        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    @pytest.mark.asyncio
    async def test_async_no_role_when_prompts_disabled(self, mock_should_send_prompts):
        """Test that no role attributes are set when prompts are disabled"""
        mock_should_send_prompts.return_value = False

        args = ["Hello, world!"]
        await set_input_attributes(self.mock_span, args)

        # Verify no attributes are set when prompts are disabled
        assert len(self.span_attributes) == 0

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_no_role_when_prompts_disabled(self, mock_should_send_prompts):
        """Test that no role attributes are set when prompts are disabled in sync function"""
        mock_should_send_prompts.return_value = False

        args = ["Hello, world!"]
        set_input_attributes_sync(self.mock_span, args)

        # Verify no attributes are set when prompts are disabled
        assert len(self.span_attributes) == 0

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    @pytest.mark.asyncio
    async def test_async_no_role_when_span_not_recording(self, mock_should_send_prompts):
        """Test that no role attributes are set when span is not recording"""
        mock_should_send_prompts.return_value = True
        self.mock_span.is_recording.return_value = False

        args = ["Hello, world!"]
        await set_input_attributes(self.mock_span, args)

        # Verify no attributes are set when span is not recording
        assert len(self.span_attributes) == 0

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_no_role_when_span_not_recording(self, mock_should_send_prompts):
        """Test that no role attributes are set when span is not recording in sync function"""
        mock_should_send_prompts.return_value = True
        self.mock_span.is_recording.return_value = False

        args = ["Hello, world!"]
        set_input_attributes_sync(self.mock_span, args)

        # Verify no attributes are set when span is not recording
        assert len(self.span_attributes) == 0

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    @pytest.mark.asyncio
    async def test_async_role_attribute_for_image_content(self, mock_should_send_prompts):
        """Test that role='user' is set when processing arguments that contain images"""
        mock_should_send_prompts.return_value = True

        # Create a mock image-like object (simplified for testing)
        mock_image_part = Mock()
        mock_image_part.mime_type = "image/jpeg"

        # Test with mixed content including mock image
        args = [["Hello with image", mock_image_part]]
        await set_input_attributes(self.mock_span, args)

        # Verify role attribute is set even when content includes images
        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

        # Verify content is also set
        assert f"{SpanAttributes.LLM_PROMPTS}.0.content" in self.span_attributes

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_role_attribute_for_image_content(self, mock_should_send_prompts):
        """Test that role='user' is set when processing arguments that contain images in sync function"""
        mock_should_send_prompts.return_value = True

        # Create a mock image-like object (simplified for testing)
        mock_image_part = Mock()
        mock_image_part.mime_type = "image/jpeg"

        # Test with mixed content including mock image
        args = [["Hello with image", mock_image_part]]
        set_input_attributes_sync(self.mock_span, args)

        # Verify role attribute is set even when content includes images
        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in self.span_attributes
        assert self.span_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

        # Verify content is also set
        assert f"{SpanAttributes.LLM_PROMPTS}.0.content" in self.span_attributes
