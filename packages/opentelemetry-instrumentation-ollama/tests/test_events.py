"""Tests for Ollama event-based instrumentation."""

import pytest
from unittest.mock import Mock, patch

from opentelemetry.instrumentation.ollama import OllamaInstrumentor
from opentelemetry.instrumentation.ollama.events import (
    message_to_event,
    completion_to_event,
    embedding_to_event,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace import SpanKind
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

class TestOllamaEvents(TestBase):
    def setUp(self):
        super().setUp()
        OllamaInstrumentor().instrument(
            use_legacy_attributes=False,
            capture_content=True,
        )
        self.client = Mock()

    def tearDown(self):
        super().tearDown()
        OllamaInstrumentor().uninstrument()

    def test_message_to_event(self):
        """Test converting chat message to event."""
        message = {"role": "user", "content": "Hello"}
        event = message_to_event(message)

        assert event.name == "gen_ai.prompt"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.OLLAMA.value
        assert event.body["role"] == "user"
        assert event.body["content"] == "Hello"

    def test_completion_to_event(self):
        """Test converting completion response to event."""
        completion = {
            "model": "llama2",
            "response": "Hi there",
            "done": True,
        }
        event = completion_to_event(completion)

        assert event.name == "gen_ai.completion"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.OLLAMA.value
        assert event.body["model"] == "llama2"
        assert event.body["content"] == "Hi there"
        assert event.body["done"] is True

    def test_embedding_to_event(self):
        """Test converting embedding response to event."""
        embedding = {
            "model": "llama2",
            "embedding": [0.1, 0.2, 0.3],
        }
        input_text = "Test text"
        event = embedding_to_event(embedding, input_text)

        assert event.name == "gen_ai.embedding"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.OLLAMA.value
        assert event.body["model"] == "llama2"
        assert event.body["embedding"] == [0.1, 0.2, 0.3]
        assert event.body["input"] == "Test text"

    def test_chat_completion_events(self):
        """Test chat completion with event-based instrumentation."""
        messages = [{"role": "user", "content": "Hello"}]
        response = {
            "model": "llama2",
            "message": {"role": "assistant", "content": "Hi there"},
            "done": True,
        }
        self.client.chat.return_value = response

        with patch("ollama.Client", return_value=self.client):
            from ollama import Client
            client = Client()
            result = client.chat(messages=messages)

            spans = self.memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == "ollama.chat"
            assert span.kind == SpanKind.CLIENT

            events = self.event_logger.get_events()
            assert len(events) == 2  # 1 prompt event + 1 completion event
            
            prompt_event = events[0]
            assert prompt_event.name == "gen_ai.prompt"
            assert prompt_event.body["role"] == "user"
            assert prompt_event.body["content"] == "Hello"

            completion_event = events[1]
            assert completion_event.name == "gen_ai.completion"
            assert completion_event.body["model"] == "llama2"
            assert completion_event.body["content"] == "Hi there"

    def test_completion_events(self):
        """Test completion with event-based instrumentation."""
        prompt = "Hello"
        response = {
            "model": "llama2",
            "response": "Hi there",
            "done": True,
        }
        self.client.generate.return_value = response

        with patch("ollama.Client", return_value=self.client):
            from ollama import Client
            client = Client()
            result = client.generate(prompt=prompt)

            spans = self.memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == "ollama.completion"
            assert span.kind == SpanKind.CLIENT

            events = self.event_logger.get_events()
            assert len(events) == 2  # 1 prompt event + 1 completion event
            
            prompt_event = events[0]
            assert prompt_event.name == "gen_ai.prompt"
            assert prompt_event.body["role"] == "user"
            assert prompt_event.body["content"] == "Hello"

            completion_event = events[1]
            assert completion_event.name == "gen_ai.completion"
            assert completion_event.body["model"] == "llama2"
            assert completion_event.body["content"] == "Hi there"

    def test_embedding_events(self):
        """Test embeddings with event-based instrumentation."""
        prompt = "Test text"
        response = {
            "model": "llama2",
            "embedding": [0.1, 0.2, 0.3],
        }
        self.client.embeddings.return_value = response

        with patch("ollama.Client", return_value=self.client):
            from ollama import Client
            client = Client()
            result = client.embeddings(prompt=prompt)

            spans = self.memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == "ollama.embeddings"
            assert span.kind == SpanKind.CLIENT

            events = self.event_logger.get_events()
            assert len(events) == 1  # 1 embedding event
            
            embedding_event = events[0]
            assert embedding_event.name == "gen_ai.embedding"
            assert embedding_event.body["model"] == "llama2"
            assert embedding_event.body["embedding"] == [0.1, 0.2, 0.3]
            assert embedding_event.body["input"] == "Test text" 