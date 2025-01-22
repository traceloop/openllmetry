"""Test event-based tracking for Together AI instrumentation."""

from unittest.mock import Mock, patch
import pytest

from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.instrumentation.together import TogetherInstrumentor

@pytest.fixture
def mock_client():
    with patch("together.Client") as mock:
        yield mock

@pytest.fixture
def mock_tracer():
    return Mock()

@pytest.fixture
def mock_meter():
    return Mock()

@pytest.fixture
def mock_event_logger():
    return Mock()

def test_event_based_completion(mock_client, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking for completion."""
    # Initialize instrumentation with event-based mode
    instrumentor = TogetherInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
        event_logger=mock_event_logger,
    )

    # Mock response
    mock_response = Mock()
    mock_response.output = {"text": "Test completion"}
    mock_client.complete.return_value = mock_response

    # Call complete
    mock_client.complete(
        model="togethercomputer/llama-2-7b",
        prompt="Test prompt",
        max_tokens=100,
    )

    # Verify events were logged
    assert mock_event_logger.add_event.call_count == 2

    # Verify prompt event
    prompt_event = mock_event_logger.add_event.call_args_list[0][0][0]
    assert prompt_event["name"] == "gen_ai.prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.PROMPT
    assert prompt_event["attributes"][SpanAttributes.LLM_PROMPT_TEMPLATE] == "Test prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Together"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_MODEL] == "togethercomputer/llama-2-7b"

    # Verify completion event
    completion_event = mock_event_logger.add_event.call_args_list[1][0][0]
    assert completion_event["name"] == "gen_ai.completion"
    assert completion_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION
    assert completion_event["attributes"][SpanAttributes.LLM_COMPLETION] == "Test completion"
    assert completion_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Together"
    assert completion_event["attributes"][SpanAttributes.LLM_RESPONSE_MODEL] == "togethercomputer/llama-2-7b"

def test_event_based_streaming(mock_client, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking for streaming completion."""
    # Initialize instrumentation with event-based mode
    instrumentor = TogetherInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
        event_logger=mock_event_logger,
    )

    # Mock streaming response
    mock_response = [Mock(output={"text": "Test"}, done=False), Mock(output={"text": " completion"}, done=True)]
    mock_client.complete_stream.return_value = iter(mock_response)

    # Call complete_stream
    response = mock_client.complete_stream(
        model="togethercomputer/llama-2-7b",
        prompt="Test prompt",
        max_tokens=100,
    )

    # Consume the stream
    list(response)

    # Verify events were logged
    assert mock_event_logger.add_event.call_count >= 3  # 1 prompt + 2 completion chunks

    # Verify prompt event
    prompt_event = mock_event_logger.add_event.call_args_list[0][0][0]
    assert prompt_event["name"] == "gen_ai.prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.PROMPT
    assert prompt_event["attributes"][SpanAttributes.LLM_PROMPT_TEMPLATE] == "Test prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_SYSTEM] == "Together"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_MODEL] == "togethercomputer/llama-2-7b"

    # Verify completion events
    completion_event1 = mock_event_logger.add_event.call_args_list[1][0][0]
    assert completion_event1["name"] == "gen_ai.completion"
    assert completion_event1["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION
    assert completion_event1["attributes"][SpanAttributes.LLM_COMPLETION] == "Test"
    assert completion_event1["attributes"][SpanAttributes.LLM_SYSTEM] == "Together"
    assert completion_event1["attributes"][SpanAttributes.LLM_RESPONSE_MODEL] == "togethercomputer/llama-2-7b"

    completion_event2 = mock_event_logger.add_event.call_args_list[2][0][0]
    assert completion_event2["name"] == "gen_ai.completion"
    assert completion_event2["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION
    assert completion_event2["attributes"][SpanAttributes.LLM_COMPLETION] == " completion"
    assert completion_event2["attributes"][SpanAttributes.LLM_SYSTEM] == "Together"
    assert completion_event2["attributes"][SpanAttributes.LLM_RESPONSE_MODEL] == "togethercomputer/llama-2-7b"