"""Tests for event-based functionality in Google Generative AI instrumentation."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor

class TestGoogleGenerativeAiEvents(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock()
        self.span = Mock()
        self.meter = Mock()
        self.event_logger = Mock()
        self.tracer_provider = Mock()
        self.meter_provider = Mock()
        self.tracer_provider.get_tracer.return_value = self.tracer
        self.meter_provider.get_meter.return_value = self.meter
        self.tracer.start_span.return_value = self.span
        self.span.__enter__ = Mock()
        self.span.__exit__ = Mock()

        self.mock_response = Mock()
        self.mock_response.text = "Test response"
        self.mock_response.usage_metadata = Mock()
        self.mock_response.usage_metadata.total_token_count = 100
        self.mock_response.usage_metadata.candidates_token_count = 50
        self.mock_response.usage_metadata.prompt_token_count = 50

        # Add safety ratings
        safety_rating = Mock()
        safety_rating.category = "HARM_CATEGORY_HARASSMENT"
        safety_rating.probability = "NEGLIGIBLE"
        self.mock_response.safety_ratings = [safety_rating]

        self.instrumentor = GoogleGenerativeAiInstrumentor()
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def tearDown(self):
        self.instrumentor.uninstrument()

    def test_event_based_completion(self):
        """Test event-based tracking for completion requests."""
        with patch("google.generativeai.generative_models.GenerativeModel") as mock_model:
            mock_model._model_name = "publishers/google/models/gemini-pro"
            mock_model.generate_content.return_value = self.mock_response

            model = mock_model()
            model.generate_content(
                "Test prompt",
                event_logger=self.event_logger,
                use_legacy_attributes=False,
            )

            # Verify prompt event
            self.event_logger.emit_event.assert_any_call({
                "name": "prompt",
                "attributes": {
                    "content": "Test prompt",
                    "role": "user",
                    "content_type": "text",
                    "model": "gemini-pro",
                },
            })

            # Verify completion event with safety attributes
            self.event_logger.emit_event.assert_any_call({
                "name": "completion",
                "attributes": {
                    "completion": "Test response",
                    "model": "gemini-pro",
                    "completion_tokens": 50,
                    "role": "assistant",
                    "content_type": "text",
                    "safety_attributes": {
                        "harm_category_harassment": "NEGLIGIBLE"
                    },
                },
            })

    def test_multi_modal_content(self):
        """Test event-based tracking for multi-modal content."""
        with patch("google.generativeai.generative_models.GenerativeModel") as mock_model:
            mock_model._model_name = "publishers/google/models/gemini-pro-vision"
            mock_model.generate_content.return_value = self.mock_response

            # Create a mock image
            mock_image = {
                "mime_type": "image/jpeg",
                "data": b"fake_image_data"
            }

            model = mock_model()
            model.generate_content(
                {"text": "Describe this image", "image": mock_image},
                event_logger=self.event_logger,
                use_legacy_attributes=False,
            )

            # Verify prompt event with multi-modal content
            self.event_logger.emit_event.assert_any_call({
                "name": "prompt",
                "attributes": {
                    "content": {"text": "Describe this image", "image": mock_image},
                    "role": "user",
                    "content_type": "multimodal",
                    "model": "gemini-pro-vision",
                },
            })

    def test_tool_calls(self):
        """Test event-based tracking for tool calls."""
        with patch("google.generativeai.generative_models.GenerativeModel") as mock_model:
            mock_model._model_name = "publishers/google/models/gemini-pro"
            
            # Create mock tool call
            tool_call = Mock()
            tool_call.name = "test_tool"
            tool_call.args = {"param": "value"}
            tool_call.response = {"result": "success"}
            
            self.mock_response.tool_calls = [tool_call]
            mock_model.generate_content.return_value = self.mock_response

            model = mock_model()
            model.generate_content(
                "Use the test tool",
                event_logger=self.event_logger,
                use_legacy_attributes=False,
            )

            # Verify tool call event
            self.event_logger.emit_event.assert_any_call({
                "name": "tool_call",
                "attributes": {
                    "llm.tool.name": "test_tool",
                    "llm.tool.input": "{'param': 'value'}",
                    "llm.tool.output": "{'result': 'success'}",
                    "llm.request.type": "tool",
                    "llm.system": "Google Generative AI",
                    "llm.request.model": "gemini-pro",
                },
            })

    def test_function_calls(self):
        """Test event-based tracking for function calls."""
        with patch("google.generativeai.generative_models.GenerativeModel") as mock_model:
            mock_model._model_name = "publishers/google/models/gemini-pro"
            
            # Create mock function call
            function_call = Mock()
            function_call.name = "test_function"
            function_call.args = {"param": "value"}
            function_call.response = {"result": "success"}
            
            self.mock_response.function_call = function_call
            mock_model.generate_content.return_value = self.mock_response

            model = mock_model()
            model.generate_content(
                "Call the test function",
                event_logger=self.event_logger,
                use_legacy_attributes=False,
            )

            # Verify function call event
            self.event_logger.emit_event.assert_any_call({
                "name": "function_call",
                "attributes": {
                    "llm.function.name": "test_function",
                    "llm.function.args": "{'param': 'value'}",
                    "llm.function.output": "{'result': 'success'}",
                    "llm.request.type": "function",
                    "llm.system": "Google Generative AI",
                    "llm.request.model": "gemini-pro",
                },
            })

    def test_event_based_chat(self):
        """Test event-based tracking for chat requests."""
        with patch("google.generativeai.generative_models.ChatSession") as mock_chat:
            mock_chat._model_name = "publishers/google/models/gemini-pro"
            mock_chat.send_message.return_value = self.mock_response

            chat = mock_chat()
            chat.send_message(
                "Test message",
                event_logger=self.event_logger,
                use_legacy_attributes=False,
            )

            # Verify prompt event
            self.event_logger.emit_event.assert_any_call({
                "name": "prompt",
                "attributes": {
                    "content": "Test message",
                    "role": "user",
                    "content_type": "text",
                    "model": "gemini-pro",
                },
            })

            # Verify completion event with safety attributes
            self.event_logger.emit_event.assert_any_call({
                "name": "completion",
                "attributes": {
                    "completion": "Test response",
                    "model": "gemini-pro",
                    "completion_tokens": 50,
                    "role": "assistant",
                    "content_type": "text",
                    "safety_attributes": {
                        "harm_category_harassment": "NEGLIGIBLE"
                    },
                },
            })

    def test_streaming_response(self):
        """Test event-based tracking for streaming responses."""
        def mock_stream():
            responses = [Mock(), Mock()]
            for r in responses:
                r.text = "partial response"
                yield r

        with patch("google.generativeai.generative_models.GenerativeModel") as mock_model:
            mock_model._model_name = "publishers/google/models/gemini-pro"
            mock_model.generate_content.return_value = mock_stream()

            model = mock_model()
            response = model.generate_content(
                "Test prompt",
                event_logger=self.event_logger,
                use_legacy_attributes=False,
            )

            # Consume the stream
            complete_response = ""
            for item in response:
                complete_response += item.text

            # Verify prompt event
            self.event_logger.emit_event.assert_any_call({
                "name": "prompt",
                "attributes": {
                    "content": "Test prompt",
                    "role": "user",
                    "content_type": "text",
                    "model": "gemini-pro",
                },
            })

            # Verify completion event
            self.event_logger.emit_event.assert_any_call({
                "name": "completion",
                "attributes": {
                    "completion": complete_response,
                    "model": "gemini-pro",
                    "role": "assistant",
                    "content_type": "text",
                },
            })

    def test_legacy_mode(self):
        """Test that legacy mode functions correctly."""
        with patch("google.generativeai.generative_models.GenerativeModel") as mock_model:
            mock_model._model_name = "publishers/google/models/gemini-pro"
            mock_model.generate_content.return_value = self.mock_response

            model = mock_model()
            model.generate_content(
                "Test prompt",
                event_logger=self.event_logger,
                use_legacy_attributes=True,
            )

            # Verify span attributes were set
            self.span.set_attribute.assert_any_call(
                f"{SpanAttributes.LLM_PROMPTS}.0.user",
                "Test prompt",
            )
            self.span.set_attribute.assert_any_call(
                SpanAttributes.LLM_REQUEST_MODEL,
                "gemini-pro",
            )
            self.span.set_attribute.assert_any_call(
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                "Test response",
            )

    def test_error_handling(self):
        """Test error handling in event-based mode."""
        with patch("google.generativeai.generative_models.GenerativeModel") as mock_model:
            mock_model._model_name = "publishers/google/models/gemini-pro"
            mock_model.generate_content.side_effect = Exception("Test error")

            model = mock_model()
            with self.assertRaises(Exception):
                model.generate_content(
                    "Test prompt",
                    event_logger=self.event_logger,
                    use_legacy_attributes=False,
                )

            # Verify prompt event was still emitted
            self.event_logger.emit_event.assert_called_with({
                "name": "prompt",
                "attributes": {
                    "content": "Test prompt",
                    "role": "user",
                    "content_type": "text",
                    "model": "gemini-pro",
                },
            })

            # Verify span status was set to error
            self.span.set_status.assert_called_with(StatusCode.ERROR)

if __name__ == "__main__":
    unittest.main() 
 