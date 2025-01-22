"""Tests for Mistral AI event-based instrumentation."""

import pytest
from unittest.mock import Mock, patch

from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor
from opentelemetry.instrumentation.mistralai.events import (
    message_to_event,
    choice_to_event,
    embedding_to_event,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace import SpanKind
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from mistralai.models.chat_completion import ChatMessage, ChatCompletionResponseChoice
from mistralai.models.embeddings import EmbeddingResponse, EmbeddingObject

class TestMistralAiEvents(TestBase):
    def setUp(self):
        super().setUp()
        MistralAiInstrumentor().instrument(
            use_legacy_attributes=False,
            capture_content=True,
        )
        self.client = Mock()

    def tearDown(self):
        super().tearDown()
        MistralAiInstrumentor().uninstrument()

    def test_message_to_event(self):
        """Test converting chat message to event."""
        message = ChatMessage(role="user", content="Hello")
        event = message_to_event(message)

        assert event.name == "gen_ai.prompt"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.MISTRAL.value
        assert event.body["role"] == "user"
        assert event.body["content"] == "Hello"

    def test_choice_to_event(self):
        """Test converting completion choice to event."""
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hi there"),
            finish_reason="stop"
        )
        event = choice_to_event(choice)

        assert event.name == "gen_ai.choice"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.MISTRAL.value
        assert event.body["index"] == 0
        assert event.body["finish_reason"] == "stop"
        assert event.body["message"]["role"] == "assistant"
        assert event.body["message"]["content"] == "Hi there"

    def test_embedding_to_event(self):
        """Test converting embedding response to event."""
        embedding = EmbeddingObject(
            embedding=[0.1, 0.2, 0.3],
            index=0,
        )
        input_text = "Test text"
        event = embedding_to_event(embedding, input_text)

        assert event.name == "gen_ai.embedding"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.MISTRAL.value
        assert event.body["index"] == 0
        assert event.body["embedding"] == [0.1, 0.2, 0.3]
        assert event.body["input"] == "Test text"

    def test_chat_completion_events(self):
        """Test chat completion with event-based instrumentation."""
        messages = [ChatMessage(role="user", content="Hello")]
        response = Mock(
            model="mistral-tiny",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi there"),
                    finish_reason="stop"
                )
            ]
        )
        self.client.chat.return_value = response

        with patch("mistralai.client.MistralClient", return_value=self.client):
            from mistralai.client import MistralClient
            client = MistralClient("test-api-key")
            result = client.chat(messages=messages)

            spans = self.memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == "mistralai.chat"
            assert span.kind == SpanKind.CLIENT

            events = self.event_logger.get_events()
            assert len(events) == 2  # 1 prompt event + 1 choice event
            
            prompt_event = events[0]
            assert prompt_event.name == "gen_ai.prompt"
            assert prompt_event.body["role"] == "user"
            assert prompt_event.body["content"] == "Hello"

            choice_event = events[1]
            assert choice_event.name == "gen_ai.choice"
            assert choice_event.body["message"]["role"] == "assistant"
            assert choice_event.body["message"]["content"] == "Hi there"

    def test_embedding_events(self):
        """Test embeddings with event-based instrumentation."""
        input_text = "Test text"
        response = Mock(
            model="mistral-embed",
            data=[
                EmbeddingObject(
                    embedding=[0.1, 0.2, 0.3],
                    index=0,
                )
            ]
        )
        self.client.embeddings.return_value = response

        with patch("mistralai.client.MistralClient", return_value=self.client):
            from mistralai.client import MistralClient
            client = MistralClient("test-api-key")
            result = client.embeddings(input=input_text)

            spans = self.memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == "mistralai.embeddings"
            assert span.kind == SpanKind.CLIENT

            events = self.event_logger.get_events()
            assert len(events) == 1  # 1 embedding event
            
            embedding_event = events[0]
            assert embedding_event.name == "gen_ai.embedding"
            assert embedding_event.body["index"] == 0
            assert embedding_event.body["embedding"] == [0.1, 0.2, 0.3]
            assert embedding_event.body["input"] == "Test text" 