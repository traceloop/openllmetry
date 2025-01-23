"""Tests for SageMaker event-based instrumentation."""

import pytest
from unittest.mock import Mock, patch

from opentelemetry.instrumentation.sagemaker import SageMakerInstrumentor
from opentelemetry.instrumentation.sagemaker.events import (
    prompt_to_event,
    completion_to_event,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace import SpanKind
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

class TestSageMakerEvents(TestBase):
    def setUp(self):
        super().setUp()
        SageMakerInstrumentor().instrument(
            use_legacy_attributes=False,
            capture_content=True,
        )
        self.client = Mock()

    def tearDown(self):
        super().tearDown()
        SageMakerInstrumentor().uninstrument()

    def test_prompt_to_event(self):
        """Test converting prompt to event."""
        prompt = {"prompt": "Hello", "temperature": 0.7}
        model = "jumpstart-dft-meta-textgeneration-llama-2-7b"
        event = prompt_to_event(prompt, model)

        assert event.name == "gen_ai.prompt"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.SAGEMAKER.value
        assert event.body["model"] == model
        assert event.body["prompt"] == prompt

    def test_completion_to_event(self):
        """Test converting completion response to event."""
        completion = {"generated_text": "Hi there!"}
        model = "jumpstart-dft-meta-textgeneration-llama-2-7b"
        event = completion_to_event(completion, model)

        assert event.name == "gen_ai.completion"
        assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.SAGEMAKER.value
        assert event.body["model"] == model
        assert event.body["completion"] == completion

    def test_invoke_endpoint_events(self):
        """Test invoke_endpoint with event-based instrumentation."""
        model = "jumpstart-dft-meta-textgeneration-llama-2-7b"
        input_data = {"prompt": "Tell me a story"}
        response = {"generated_text": "Once upon a time..."}
        self.client.invoke_endpoint.return_value = response

        with patch("boto3.client", return_value=self.client):
            import boto3
            client = boto3.client("sagemaker-runtime")
            result = client.invoke_endpoint(
                EndpointName=model,
                ContentType="application/json",
                Body=json.dumps(input_data)
            )

            spans = self.memory_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == "sagemaker.invoke_endpoint"
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