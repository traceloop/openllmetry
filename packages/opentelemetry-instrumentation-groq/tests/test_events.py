"""Tests for Groq instrumentation event-based functionality."""

import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from opentelemetry.instrumentation.groq import GroqInstrumentor
from opentelemetry.instrumentation.groq.events import (
    create_prompt_event,
    create_completion_event,
    create_tool_call_event,
    create_function_call_event,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, StatusCode


class TestGroqEvents(unittest.TestCase):
    """Test cases for Groq event-based instrumentation."""

    def setUp(self):
        self.tracer_provider = Mock()
        self.tracer = Mock()
        self.span = MagicMock()
        self.meter_provider = Mock()
        self.meter = Mock()
        self.event_logger = Mock()
        self.mock_client = Mock()

        self.tracer_provider.get_tracer.return_value = self.tracer
        self.tracer.start_span.return_value = self.span
        self.span.__enter__ = Mock()
        self.span.__exit__ = Mock()

        self.instrumentor = GroqInstrumentor()
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def tearDown(self):
        self.instrumentor.uninstrument()

    def test_event_based_completion(self):
        """Test event-based tracking for completion requests."""
        response = {
            "model": "mixtral-8x7b-32768",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test completion",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

        self.mock_client.create.return_value = response

        with patch("groq.resources.chat.completions.Completions.create", self.mock_client.create):
            self.mock_client.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": "Test prompt"}],
                event_logger=self.event_logger,
                use_legacy_attributes=False,
            )

        # Verify prompt event
        self.event_logger.emit_event.assert_any_call({
            "name": "llm.prompts",
            "attributes": {
                "llm.prompts.content": "Test prompt",
                "llm.prompts.role": "user",
                "llm.prompts.content_type": "text",
                "llm.model": "mixtral-8x7b-32768",
            },
        })

        # Verify completion event
        self.event_logger.emit_event.assert_any_call({
            "name": "llm.completions",
            "attributes": {
                "llm.completions.content": "Test completion",
                "llm.completions.role": "assistant",
                "llm.completions.finish_reason": "stop",
                "llm.model": "mixtral-8x7b-32768",
            },
        })

    def test_event_based_chat(self):
        """Test event-based tracking for chat requests."""
        response = {
            "model": "mixtral-8x7b-32768",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test chat response",
                        "function_call": {
                            "name": "test_function",
                            "arguments": json.dumps({"arg": "value"}),
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
        }

        self.mock_client.create.return_value = response

        with patch("groq.resources.chat.completions.Completions.create", self.mock_client.create):
            self.mock_client.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "System message"},
                    {"role": "user", "content": "User message"},
                ],
                event_logger=self.event_logger,
                use_legacy_attributes=False,
            )

        # Verify system prompt event
        self.event_logger.emit_event.assert_any_call({
            "name": "llm.prompts",
            "attributes": {
                "llm.prompts.content": "System message",
                "llm.prompts.role": "system",
                "llm.prompts.content_type": "text",
                "llm.model": "mixtral-8x7b-32768",
            },
        })

        # Verify user prompt event
        self.event_logger.emit_event.assert_any_call({
            "name": "llm.prompts",
            "attributes": {
                "llm.prompts.content": "User message",
                "llm.prompts.role": "user",
                "llm.prompts.content_type": "text",
                "llm.model": "mixtral-8x7b-32768",
            },
        })

        # Verify completion event
        self.event_logger.emit_event.assert_any_call({
            "name": "llm.completions",
            "attributes": {
                "llm.completions.content": "Test chat response",
                "llm.completions.role": "assistant",
                "llm.completions.finish_reason": "function_call",
                "llm.model": "mixtral-8x7b-32768",
            },
        })

        # Verify function call event
        self.event_logger.emit_event.assert_any_call({
            "name": "llm.function.calls",
            "attributes": {
                "llm.function.name": "test_function",
                "llm.function.args": json.dumps({"arg": "value"}),
                "llm.model": "mixtral-8x7b-32768",
            },
        })

    def test_streaming_response(self):
        """Test event-based tracking for streaming responses."""
        class MockStream:
            def __iter__(self):
                yield {
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": "Test",
                            },
                            "finish_reason": None,
                        }
                    ]
                }
                yield {
                    "choices": [
                        {
                            "delta": {
                                "content": " streaming",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }

        self.mock_client.create.return_value = MockStream()

        with patch("groq.resources.chat.completions.Completions.create", self.mock_client.create):
            self.mock_client.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": "Test prompt"}],
                stream=True,
                event_logger=self.event_logger,
                use_legacy_attributes=False,
            )

        # Verify prompt event
        self.event_logger.emit_event.assert_any_call({
            "name": "llm.prompts",
            "attributes": {
                "llm.prompts.content": "Test prompt",
                "llm.prompts.role": "user",
                "llm.prompts.content_type": "text",
                "llm.model": "mixtral-8x7b-32768",
            },
        })

        # Verify completion event
        self.event_logger.emit_event.assert_any_call({
            "name": "llm.completions",
            "attributes": {
                "llm.completions.content": "Test streaming",
                "llm.completions.role": "assistant",
                "llm.completions.finish_reason": "stop",
                "llm.model": "mixtral-8x7b-32768",
            },
        })

    def test_legacy_mode(self):
        """Test legacy mode functionality."""
        response = {
            "model": "mixtral-8x7b-32768",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test completion",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

        self.mock_client.create.return_value = response

        with patch("groq.resources.chat.completions.Completions.create", self.mock_client.create):
            self.mock_client.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": "Test prompt"}],
                event_logger=self.event_logger,
                use_legacy_attributes=True,
            )

        # Verify span attributes are set in legacy mode
        self.span.set_attribute.assert_any_call(
            SpanAttributes.LLM_REQUEST_MODEL,
            "mixtral-8x7b-32768",
        )
        self.span.set_attribute.assert_any_call(
            f"{SpanAttributes.LLM_PROMPTS}.0.content",
            "Test prompt",
        )
        self.span.set_attribute.assert_any_call(
            f"{SpanAttributes.LLM_PROMPTS}.0.role",
            "user",
        )
        self.span.set_attribute.assert_any_call(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
            "Test completion",
        )
        self.span.set_attribute.assert_any_call(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.role",
            "assistant",
        )
        self.span.set_attribute.assert_any_call(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason",
            "stop",
        )

    def test_error_handling(self):
        """Test error handling in event-based mode."""
        error_message = "Test error"
        self.mock_client.create.side_effect = Exception(error_message)

        with patch("groq.resources.chat.completions.Completions.create", self.mock_client.create):
            with self.assertRaises(Exception):
                self.mock_client.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": "Test prompt"}],
                    event_logger=self.event_logger,
                    use_legacy_attributes=False,
                )

        # Verify prompt event is emitted before error
        self.event_logger.emit_event.assert_called_with({
            "name": "llm.prompts",
            "attributes": {
                "llm.prompts.content": "Test prompt",
                "llm.prompts.role": "user",
                "llm.prompts.content_type": "text",
                "llm.model": "mixtral-8x7b-32768",
            },
        })

        # Verify span status is set to error
        self.span.set_status.assert_called_with(StatusCode.ERROR)
        self.span.set_status.assert_called_with(StatusCode.ERROR, error_message) 
 
 
 