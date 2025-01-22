"""Test event-based functionality for Bedrock instrumentation."""

import json
from unittest.mock import Mock, patch

import pytest
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues


@pytest.fixture
def mock_client():
    """Create a mock Bedrock client."""
    with patch("botocore.client.ClientCreator") as mock_creator:
        mock_client = Mock()
        mock_creator.create_client.return_value = mock_client
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


def test_event_based_mode(mock_client, mock_tracer, mock_meter, mock_event_logger):
    """Test that event-based mode works correctly."""
    # Initialize instrumentation with event-based mode
    instrumentor = BedrockInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
        event_logger=mock_event_logger,
    )

    # Mock response for invoke_model
    mock_response = {
        "body": Mock(
            read=lambda: json.dumps(
                {
                    "completion": "Test completion",
                    "stop_reason": "stop",
                }
            )
        )
    }
    mock_client.invoke_model.return_value = mock_response

    # Call invoke_model with test data
    mock_client.invoke_model(
        modelId="anthropic.claude-v2",
        body=json.dumps({"prompt": "Test prompt"}),
    )

    # Verify events were logged
    assert mock_event_logger.add_event.call_count == 2

    # Verify prompt event
    prompt_event = mock_event_logger.add_event.call_args_list[0][0][0]
    assert prompt_event["name"] == "prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.PROMPT
    assert prompt_event["attributes"][SpanAttributes.LLM_PROMPT_TEMPLATE] == "Test prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_SYSTEM] == "anthropic"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_MODEL] == "claude-v2"

    # Verify completion event
    completion_event = mock_event_logger.add_event.call_args_list[1][0][0]
    assert completion_event["name"] == "completion"
    assert completion_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION
    assert completion_event["attributes"][SpanAttributes.LLM_COMPLETION] == "Test completion"
    assert completion_event["attributes"][SpanAttributes.LLM_SYSTEM] == "anthropic"
    assert completion_event["attributes"][SpanAttributes.LLM_RESPONSE_MODEL] == "claude-v2"
    assert completion_event["attributes"][SpanAttributes.LLM_RESPONSE_FINISH_REASON] == "stop"


def test_legacy_mode(mock_client, mock_tracer, mock_meter):
    """Test that legacy mode works correctly."""
    # Initialize instrumentation with legacy mode
    instrumentor = BedrockInstrumentor(use_legacy_attributes=True)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
    )

    # Mock response for invoke_model
    mock_response = {
        "body": Mock(
            read=lambda: json.dumps(
                {
                    "completion": "Test completion",
                    "stop_reason": "stop",
                }
            )
        )
    }
    mock_client.invoke_model.return_value = mock_response

    # Call invoke_model with test data
    mock_client.invoke_model(
        modelId="anthropic.claude-v2",
        body=json.dumps({"prompt": "Test prompt"}),
    )

    # Verify span attributes were set
    mock_span = mock_tracer.start_span.return_value
    mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_SYSTEM, "anthropic")
    mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_REQUEST_MODEL, "claude-v2")
    mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_RESPONSE_MODEL, "claude-v2")
    mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_PROMPT_TEMPLATE, "Test prompt")
    mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_COMPLETION, "Test completion")


def test_streaming_events(mock_client, mock_tracer, mock_meter, mock_event_logger):
    """Test that streaming events work correctly."""
    # Initialize instrumentation with event-based mode
    instrumentor = BedrockInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
        event_logger=mock_event_logger,
    )

    # Mock streaming response
    mock_stream = Mock()
    mock_stream._raw_stream = iter([
        json.dumps({
            "completion": "Test streaming completion",
            "stop_reason": "stop",
        }).encode()
    ])
    mock_stream._content_length = 100

    mock_response = {
        "body": mock_stream
    }
    mock_client.invoke_model_with_response_stream.return_value = mock_response

    # Call invoke_model_with_response_stream with test data
    mock_client.invoke_model_with_response_stream(
        modelId="anthropic.claude-v2",
        body=json.dumps({"prompt": "Test streaming prompt"}),
    )

    # Verify events were logged
    assert mock_event_logger.add_event.call_count == 2

    # Verify prompt event
    prompt_event = mock_event_logger.add_event.call_args_list[0][0][0]
    assert prompt_event["name"] == "prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.PROMPT
    assert prompt_event["attributes"][SpanAttributes.LLM_PROMPT_TEMPLATE] == "Test streaming prompt"
    assert prompt_event["attributes"][SpanAttributes.LLM_SYSTEM] == "anthropic"
    assert prompt_event["attributes"][SpanAttributes.LLM_REQUEST_MODEL] == "claude-v2"

    # Verify completion event
    completion_event = mock_event_logger.add_event.call_args_list[1][0][0]
    assert completion_event["name"] == "completion"
    assert completion_event["attributes"][SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION
    assert completion_event["attributes"][SpanAttributes.LLM_COMPLETION] == "Test streaming completion"
    assert completion_event["attributes"][SpanAttributes.LLM_SYSTEM] == "anthropic"
    assert completion_event["attributes"][SpanAttributes.LLM_RESPONSE_MODEL] == "claude-v2"
    assert completion_event["attributes"][SpanAttributes.LLM_RESPONSE_FINISH_REASON] == "stop"


def test_error_handling(mock_client, mock_tracer, mock_meter, mock_event_logger):
    """Test error handling in event-based mode."""
    # Initialize instrumentation with event-based mode
    instrumentor = BedrockInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=Mock(),
        meter_provider=Mock(),
        event_logger=mock_event_logger,
    )

    # Mock error response
    class MockError(Exception):
        pass

    mock_client.invoke_model.side_effect = MockError("Test error")

    # Call invoke_model with test data and expect error
    with pytest.raises(MockError):
        mock_client.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps({"prompt": "Test prompt"}),
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