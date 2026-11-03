import json
import pytest
from unittest.mock import Mock, patch
from opentelemetry.instrumentation.vertexai.span_utils import (
    set_input_attributes,
    set_input_attributes_sync,
)
from opentelemetry.semconv_ai import SpanAttributes
from vertexai.generative_models import Content, Part


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


class TestContentObjectHistory:
    """Regression tests for issue #2513: a list of Content objects is a
    conversation history, so every Content must become its own prompt entry
    with its own role instead of being merged into a single user prompt."""

    def setup_method(self):
        self.mock_span = Mock()
        self.mock_span.is_recording.return_value = True
        self.mock_span.context.trace_id = "test_trace_id"
        self.mock_span.context.span_id = "test_span_id"
        self.span_attributes = {}

        def capture_attribute(key, value):
            self.span_attributes[key] = value

        self.mock_span.set_attribute = capture_attribute

    @staticmethod
    def _history_args():
        return [
            [
                Content(role="user", parts=[Part.from_text("What's 2+2?")]),
                Content(role="model", parts=[Part.from_text("5")]),
                Content(role="user", parts=[Part.from_text("really?")]),
            ]
        ]

    def _assert_history_attributes(self):
        prompts = SpanAttributes.LLM_PROMPTS
        assert self.span_attributes[f"{prompts}.0.role"] == "user"
        assert json.loads(self.span_attributes[f"{prompts}.0.content"]) == [
            {"type": "text", "text": "What's 2+2?"}
        ]
        # VertexAI's "model" role is normalized to "assistant"
        assert self.span_attributes[f"{prompts}.1.role"] == "assistant"
        assert json.loads(self.span_attributes[f"{prompts}.1.content"]) == [
            {"type": "text", "text": "5"}
        ]
        assert self.span_attributes[f"{prompts}.2.role"] == "user"
        assert json.loads(self.span_attributes[f"{prompts}.2.content"]) == [
            {"type": "text", "text": "really?"}
        ]
        assert f"{prompts}.3.role" not in self.span_attributes

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_content_history_preserves_roles(self, mock_should_send_prompts):
        mock_should_send_prompts.return_value = True

        set_input_attributes_sync(self.mock_span, self._history_args())

        self._assert_history_attributes()

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    @pytest.mark.asyncio
    async def test_async_content_history_preserves_roles(
        self, mock_should_send_prompts
    ):
        mock_should_send_prompts.return_value = True

        await set_input_attributes(self.mock_span, self._history_args())

        self._assert_history_attributes()

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_single_content_argument_keeps_role(self, mock_should_send_prompts):
        mock_should_send_prompts.return_value = True

        args = [Content(role="model", parts=[Part.from_text("an answer")])]
        set_input_attributes_sync(self.mock_span, args)

        prompts = SpanAttributes.LLM_PROMPTS
        assert self.span_attributes[f"{prompts}.0.role"] == "assistant"
        assert json.loads(self.span_attributes[f"{prompts}.0.content"]) == [
            {"type": "text", "text": "an answer"}
        ]

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_multi_part_content_stays_one_message(
        self, mock_should_send_prompts
    ):
        mock_should_send_prompts.return_value = True

        args = [
            [
                Content(
                    role="user",
                    parts=[Part.from_text("first"), Part.from_text("second")],
                )
            ]
        ]
        set_input_attributes_sync(self.mock_span, args)

        prompts = SpanAttributes.LLM_PROMPTS
        assert self.span_attributes[f"{prompts}.0.role"] == "user"
        assert json.loads(self.span_attributes[f"{prompts}.0.content"]) == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert f"{prompts}.1.role" not in self.span_attributes

    @patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts")
    def test_sync_parts_list_stays_one_user_message(self, mock_should_send_prompts):
        """A list of parts (not Content objects) is a single user message."""
        mock_should_send_prompts.return_value = True

        args = [[Part.from_text("hello"), "world"]]
        set_input_attributes_sync(self.mock_span, args)

        prompts = SpanAttributes.LLM_PROMPTS
        assert self.span_attributes[f"{prompts}.0.role"] == "user"
        assert json.loads(self.span_attributes[f"{prompts}.0.content"]) == [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert f"{prompts}.1.role" not in self.span_attributes
