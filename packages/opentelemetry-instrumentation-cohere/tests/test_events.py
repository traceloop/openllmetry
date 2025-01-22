"""Test event-based functionality for Cohere instrumentation."""

import json
from unittest.mock import Mock, patch

import pytest
from opentelemetry.instrumentation.cohere import CohereInstrumentor
from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues


@pytest.fixture
def mock_client():
    """Create a mock Cohere client."""
    with patch("cohere.Client") as mock_client:
        yield mock_client


@pytest.fixture
def mock_tracer():
    """Create a mock tracer."""
    with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer
        yield mock_tracer


@pytest.fixture
def mock_meter():
    """Create a mock meter."""
    with patch("opentelemetry.metrics.get_meter") as mock_get_meter:
        mock_meter = Mock()
        mock_get_meter.return_value = mock_meter
        yield mock_meter


@pytest.fixture
def mock_event_logger():
    """Create a mock event logger."""
    mock_logger = Mock()
    return mock_logger


def test_event_based_completion(mock_client, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking for completion."""
    # Initialize instrumentation with event-based mode
    instrumentor = CohereInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
        event_logger=mock_event_logger,
    )

    # Mock response
    mock_response = Mock()
    mock_response.generations = [Mock(text="Test completion")]
    mock_client.generate.return_value = mock_response

    # Call generate
    mock_client.generate(
        model="command",
        prompt="Test prompt",
        max_tokens=100,
    )

    # Verify events were logged
    assert mock_event_logger.add_event.call_count == 2

    # Verify prompt event
    prompt_event = mock_event_logger.add_event.call_args_list[0][0][0]
    assert prompt_event["name"] == "prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.PROMPT
    assert prompt_event["attributes"][SpanAttributes.LLM_PROMPT_TEMPLATE] == "Test prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Cohere"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_MODEL] == "command"

    # Verify completion event
    completion_event = mock_event_logger.add_event.call_args_list[1][0][0]
    assert completion_event["name"] == "completion"
    assert completion_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION
    assert completion_event["attributes"][SpanAttributes.LLM_COMPLETION] == "Test completion"
    assert completion_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Cohere"
    assert completion_event["attributes"][SpanAttributes.LLM_RESPONSE_MODEL] == "command"


def test_event_based_chat(mock_client, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking for chat."""
    # Initialize instrumentation with event-based mode
    instrumentor = CohereInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
        event_logger=mock_event_logger,
    )

    # Mock response
    mock_response = Mock()
    mock_response.text = "Test chat response"
    mock_response.token_count = {
        "response_tokens": 10,
    }
    mock_client.chat.return_value = mock_response

    # Call chat
    mock_client.chat(
        model="command",
        message="Test message",
    )

    # Verify events were logged
    assert mock_event_logger.add_event.call_count == 2

    # Verify prompt event
    prompt_event = mock_event_logger.add_event.call_args_list[0][0][0]
    assert prompt_event["name"] == "prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.PROMPT
    assert prompt_event["attributes"][SpanAttributes.LLM_PROMPT_TEMPLATE] == "Test message"
    assert prompt_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Cohere"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_MODEL] == "command"

    # Verify completion event
    completion_event = mock_event_logger.add_event.call_args_list[1][0][0]
    assert completion_event["name"] == "completion"
    assert completion_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION
    assert completion_event["attributes"][SpanAttributes.LLM_COMPLETION] == "Test chat response"
    assert completion_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Cohere"
    assert completion_event["attributes"][SpanAttributes.LLM_RESPONSE_MODEL] == "command"
    assert completion_event["attributes"][SpanAttributes.LLM_TOKEN_COUNT] == 10


def test_event_based_rerank(mock_client, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking for rerank."""
    # Initialize instrumentation with event-based mode
    instrumentor = CohereInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
        event_logger=mock_event_logger,
    )

    # Mock response
    mock_doc = Mock()
    mock_doc.index = 0
    mock_doc.relevance_score = 0.9
    mock_doc.document = Mock(text="Test document")

    mock_response = Mock()
    mock_response.results = [mock_doc]
    mock_client.rerank.return_value = mock_response

    # Call rerank
    mock_client.rerank(
        model="rerank-english-v2.0",
        documents=["Test document"],
        query="Test query",
    )

    # Verify events were logged (1 system prompt + 1 user prompt + 1 completion)
    assert mock_event_logger.add_event.call_count == 3

    # Verify document prompt event
    doc_event = mock_event_logger.add_event.call_args_list[0][0][0]
    assert doc_event["name"] == "prompt"
    assert doc_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.PROMPT
    assert doc_event["attributes"][SpanAttributes.LLM_PROMPT_TEMPLATE] == "Test document"
    assert doc_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Cohere"
    assert doc_event["attributes"][SpanAttributes.LLM_REQUEST_MODEL] == "rerank-english-v2.0"
    assert doc_event["attributes"][SpanAttributes.LLM_REQUEST_ROLE] == "system"

    # Verify query prompt event
    query_event = mock_event_logger.add_event.call_args_list[1][0][0]
    assert query_event["name"] == "prompt"
    assert query_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.PROMPT
    assert query_event["attributes"][SpanAttributes.LLM_PROMPT_TEMPLATE] == "Test query"
    assert query_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Cohere"
    assert query_event["attributes"][SpanAttributes.LLM_REQUEST_MODEL] == "rerank-english-v2.0"
    assert query_event["attributes"][SpanAttributes.LLM_REQUEST_ROLE] == "user"

    # Verify rerank result event
    result_event = mock_event_logger.add_event.call_args_list[2][0][0]
    assert result_event["name"] == "completion"
    assert result_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION
    assert "Doc 0, Score: 0.9\nTest document" in result_event["attributes"][SpanAttributes.LLM_COMPLETION]
    assert result_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Cohere"
    assert result_event["attributes"][SpanAttributes.LLM_RESPONSE_MODEL] == "rerank-english-v2.0"
    assert result_event["attributes"][SpanAttributes.LLM_RESPONSE_ROLE] == "assistant"


def test_legacy_mode(mock_client, mock_tracer, mock_meter):
    """Test that legacy mode works correctly."""
    # Initialize instrumentation with legacy mode
    instrumentor = CohereInstrumentor(use_legacy_attributes=True)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
    )

    # Mock response
    mock_response = Mock()
    mock_response.generations = [Mock(text="Test completion")]
    mock_client.generate.return_value = mock_response

    # Call generate
    mock_client.generate(
        model="command",
        prompt="Test prompt",
        max_tokens=100,
    )

    # Verify span attributes were set
    mock_span = mock_tracer.start_span.return_value
    mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_SYSTEM, "Cohere")
    mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_REQUEST_MODEL, "command")
    mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
    mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.0.content", "Test prompt")
    mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", "Test completion")


def test_error_handling(mock_client, mock_tracer, mock_meter, mock_event_logger):
    """Test error handling in event-based mode."""
    # Initialize instrumentation with event-based mode
    instrumentor = CohereInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
        event_logger=mock_event_logger,
    )

    # Mock error response
    class MockError(Exception):
        pass

    mock_client.generate.side_effect = MockError("Test error")

    # Call generate with test data and expect error
    with pytest.raises(MockError):
        mock_client.generate(
            model="command",
            prompt="Test prompt",
        )

    # Verify prompt event was logged before error
    assert mock_event_logger.add_event.call_count == 1
    prompt_event = mock_event_logger.add_event.call_args_list[0][0][0]
    assert prompt_event["name"] == "prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.PROMPT
    assert prompt_event["attributes"][SpanAttributes.LLM_PROMPT_TEMPLATE] == "Test prompt"

    # Verify span recorded error
    mock_span = mock_tracer.start_span.return_value
    mock_span.record_exception.assert_called_once() 