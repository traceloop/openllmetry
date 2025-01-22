"""Tests for Replicate event-based instrumentation."""

import pytest
from unittest.mock import Mock, patch

from opentelemetry.instrumentation.replicate import ReplicateInstrumentor
from opentelemetry.instrumentation.replicate.events import (
    prompt_to_event,
    completion_to_event,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace import SpanKind
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

class TestReplicateEvents(TestBase):
    def setUp(self):
        super().setUp()
        ReplicateInstrumentor().instrument(
            use_legacy_attributes=False,
            capture_content=True,
        )
        self.client = Mock()

    def tearDown(self):
        super().tearDown()
        ReplicateInstrumentor().uninstrument()

    def test_prompt_to_event(self):
        """Test converting prompt to event."""
        prompt = {"prompt": "Hello", "temperature": 0.7}
        model = "stability-ai/sdxl"
        event = prompt_to_event(prompt, model)

        assert event.name == "gen_ai.prompt"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.REPLICATE.value
        assert event.body["model"] == model
        assert event.body["prompt"] == prompt

    def test_completion_to_event(self):
        """Test converting completion response to event."""
        completion = ["image1.png", "image2.png"]
        model = "stability-ai/sdxl"
        event = completion_to_event(completion, model)

        assert event.name == "gen_ai.completion"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.REPLICATE.value
        assert event.body["model"] == model
        assert event.body["completion"] == completion

    def test_run_events(self):
        """Test run with event-based instrumentation."""
        model = "stability-ai/sdxl"
        input_data = {"prompt": "A beautiful sunset"}
        response = ["image1.png", "image2.png"]
        self.client.run.return_value = response

        with patch("replicate.Client", return_value=self.client):
            from replicate import Client
            client = Client()
            result = client.run(model, input=input_data)

            spans = self.memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == "replicate.run"
            assert span.kind == SpanKind.CLIENT

            events = self.event_logger.get_events()
            assert len(events) == 2  # 1 prompt event + 1 completion event
            
            prompt_event = events[0]
            assert prompt_event.name == "gen_ai.prompt"
            assert prompt_event.body["model"] == model
            assert prompt_event.body["prompt"] == input_data

            completion_event = events[1]
            assert completion_event.name == "gen_ai.completion"
            assert completion_event.body["model"] == model
            assert completion_event.body["completion"] == response

    def test_streaming_events(self):
        """Test streaming with event-based instrumentation."""
        model = "meta/llama-2-70b-chat"
        input_data = {"prompt": "Tell me a story"}
        response = ["Once", " upon", " a", " time"]
        self.client.run.return_value = response

        with patch("replicate.Client", return_value=self.client):
            from replicate import Client
            client = Client()
            result = client.run(model, input=input_data, stream=True)
            chunks = list(result)  # Consume generator

            spans = self.memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == "replicate.run"
            assert span.kind == SpanKind.CLIENT

            events = self.event_logger.get_events()
            assert len(events) == 5  # 1 prompt event + 4 completion events (one per chunk)
            
            prompt_event = events[0]
            assert prompt_event.name == "gen_ai.prompt"
            assert prompt_event.body["model"] == model
            assert prompt_event.body["prompt"] == input_data

            for i, chunk in enumerate(response):
                completion_event = events[i + 1]
                assert completion_event.name == "gen_ai.completion"
                assert completion_event.body["model"] == model
                assert completion_event.body["completion"] == chunk

    def test_predictions_create_events(self):
        """Test predictions.create with event-based instrumentation."""
        version = {
            "id": "5797a99edc939ea0e9242d5e8c9cb3bc7d125b1eac21bda852e5cb79ede2cd9b",
            "model": "kvfrans/clipdraw",
        }
        input_data = {"prompt": "A beautiful sunset"}
        response = ["image1.png"]
        self.client.predictions.create.return_value = response

        with patch("replicate.Client", return_value=self.client):
            from replicate import Client
            client = Client()
            result = client.predictions.create(version=version, input=input_data)

            spans = self.memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == "replicate.predictions.create"
            assert span.kind == SpanKind.CLIENT

            events = self.event_logger.get_events()
            assert len(events) == 2  # 1 prompt event + 1 completion event
            
            prompt_event = events[0]
            assert prompt_event.name == "gen_ai.prompt"
            assert prompt_event.body["model"] == version["model"]
            assert prompt_event.body["prompt"] == input_data

            completion_event = events[1]
            assert completion_event.name == "gen_ai.completion"
            assert completion_event.body["model"] == version["model"]
            assert completion_event.body["completion"] == response 