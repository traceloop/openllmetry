"""Test event-based tracking for WatsonX instrumentation."""

import pytest
from unittest.mock import Mock, patch
from opentelemetry.semconv._incubating.attributes import GenAIAttributes
from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor
from opentelemetry.instrumentation.watsonx.events import prompt_to_event, completion_to_event
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.span import Span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


@pytest.fixture
def mock_watsonx():
    with patch("ibm_watson_machine_learning.foundation_models.inference.ModelInference") as mock:
        instance = mock.return_value
        instance.model_id = "test-model"
        instance.generate.return_value = {
            "results": [{
                "generated_text": "Test completion",
                "input_token_count": 10,
                "generated_token_count": 20,
                "stop_reason": "max_tokens"
            }]
        }
        yield instance


@pytest.fixture
def mock_tracer():
    tracer_provider = TracerProvider()
    memory_exporter = InMemorySpanExporter()
    span_processor = SimpleSpanProcessor(memory_exporter)
    tracer_provider.add_span_processor(span_processor)
    with patch("opentelemetry.trace.get_tracer_provider") as mock:
        mock.return_value = tracer_provider
        yield tracer_provider


@pytest.fixture
def mock_meter():
    meter_provider = MeterProvider()
    reader = InMemoryMetricReader()
    meter_provider.add_reader(reader)
    with patch("opentelemetry.metrics.get_meter_provider") as mock:
        mock.return_value = meter_provider
        yield meter_provider


@pytest.fixture
def mock_event_logger():
    return Mock()


class TestWatsonxEvents:
    def setUp(self):
        self.instrumentor = WatsonxInstrumentor()
        self.instrumentor.instrument()

    def tearDown(self):
        self.instrumentor.uninstrument()

    def test_prompt_to_event(self):
        """Test converting a prompt to an event."""
        prompt = "Test prompt"
        model_name = "test-model"
        event = prompt_to_event(prompt=prompt, model_name=model_name, capture_content=True)
        
        assert event.name == "watsonx.prompt"
        assert event.attributes["llm.system"] == "watsonx"
        assert event.attributes["llm.model"] == model_name
        assert event.body == prompt

    def test_completion_to_event(self):
        """Test converting a completion to an event."""
        completion = {
            "generated_text": "Test completion",
            "token_usage": {
                "prompt_tokens": 10,
                "generated_tokens": 20
            }
        }
        model_name = "test-model"
        event = completion_to_event(completion=completion, model_name=model_name, capture_content=True)
        
        assert event.name == "watsonx.completion"
        assert event.attributes["llm.system"] == "watsonx"
        assert event.attributes["llm.model"] == model_name
        assert event.attributes["llm.usage.prompt_tokens"] == 10
        assert event.attributes["llm.usage.completion_tokens"] == 20
        assert event.body == completion["generated_text"]

    def test_event_based_completion(self, mock_watsonx, mock_tracer, mock_meter, mock_event_logger):
        """Test event-based tracking for text completion."""
        self.instrumentor._config.event_logger = mock_event_logger
        self.instrumentor._config.capture_content = True

        prompt = "Test prompt"
        mock_watsonx.generate(prompt)

        # Verify prompt event
        prompt_event_call = mock_event_logger.emit.call_args_list[0]
        prompt_event = prompt_event_call[0][0]
        assert prompt_event.name == "watsonx.prompt"
        assert prompt_event.attributes["llm.system"] == "watsonx"
        assert prompt_event.body == prompt

        # Verify completion event
        completion_event_call = mock_event_logger.emit.call_args_list[1]
        completion_event = completion_event_call[0][0]
        assert completion_event.name == "watsonx.completion"
        assert completion_event.attributes["llm.system"] == "watsonx"
        assert completion_event.attributes["llm.usage.prompt_tokens"] == 10
        assert completion_event.attributes["llm.usage.completion_tokens"] == 20
        assert completion_event.body == "Test completion"

    def test_event_based_streaming(self, mock_watsonx, mock_tracer, mock_meter, mock_event_logger):
        """Test event-based tracking for streaming responses."""
        self.instrumentor._config.event_logger = mock_event_logger
        self.instrumentor._config.capture_content = True

        prompt = "Test prompt"
        mock_watsonx.generate_text_stream.return_value = iter([
            {"results": [{
                "generated_text": "Part 1",
                "input_token_count": 5,
                "generated_token_count": 10
            }]},
            {"results": [{
                "generated_text": "Part 2",
                "input_token_count": 5,
                "generated_token_count": 10
            }]}
        ])

        list(mock_watsonx.generate_text_stream(prompt))

        # Verify prompt event
        prompt_event_call = mock_event_logger.emit.call_args_list[0]
        prompt_event = prompt_event_call[0][0]
        assert prompt_event.name == "watsonx.prompt"
        assert prompt_event.body == prompt

        # Verify completion event for streaming
        completion_event_call = mock_event_logger.emit.call_args_list[1]
        completion_event = completion_event_call[0][0]
        assert completion_event.name == "watsonx.completion"
        assert completion_event.attributes["llm.system"] == "watsonx"
        assert completion_event.attributes["llm.usage.prompt_tokens"] == 10
        assert completion_event.attributes["llm.usage.completion_tokens"] == 20
        assert completion_event.body == "Part 1Part 2"

    def test_event_based_no_content_capture(self, mock_watsonx, mock_tracer, mock_meter, mock_event_logger):
        """Test event-based tracking with content capture disabled."""
        self.instrumentor._config.event_logger = mock_event_logger
        self.instrumentor._config.capture_content = False

        prompt = "Test prompt"
        mock_watsonx.generate(prompt)

        # Verify prompt event without content
        prompt_event_call = mock_event_logger.emit.call_args_list[0]
        prompt_event = prompt_event_call[0][0]
        assert prompt_event.name == "watsonx.prompt"
        assert prompt_event.body is None

        # Verify completion event without content
        completion_event_call = mock_event_logger.emit.call_args_list[1]
        completion_event = completion_event_call[0][0]
        assert completion_event.name == "watsonx.completion"
        assert completion_event.body is None
        assert completion_event.attributes["llm.usage.prompt_tokens"] == 10
        assert completion_event.attributes["llm.usage.completion_tokens"] == 20 