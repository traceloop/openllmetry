"""Test event-based tracking for VertexAI instrumentation."""

from unittest.mock import Mock, patch
import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor

@pytest.fixture
def mock_vertexai():
    with patch("vertexai.generative_models") as mock:
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

def test_event_based_completion(mock_vertexai, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking for text completion."""
    # Initialize instrumentation with event-based mode
    instrumentor = VertexAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=Mock(get_tracer=Mock(return_value=mock_tracer)),
        meter_provider=Mock(get_meter=Mock(return_value=mock_meter)),
        event_logger=mock_event_logger,
        use_legacy_attributes=False,
    )

    # Mock response
    mock_response = Mock()
    mock_response.text = "Test completion"
    mock_response.safety_attributes = {"harmful": 0.1, "hate": 0.0}
    mock_vertexai.GenerativeModel.return_value.generate_content.return_value = mock_response

    # Create model and generate content
    model = mock_vertexai.GenerativeModel("gemini-pro")
    model.generate_content("Test prompt")

    # Verify events were logged
    assert mock_event_logger.emit.call_count == 2

    # Verify prompt event
    prompt_event = mock_event_logger.emit.call_args_list[0][0][0]
    assert prompt_event.name == "gen_ai.prompt"
    assert prompt_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.VERTEXAI.value
    assert prompt_event.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gemini-pro"
    assert prompt_event.body["content"] == "Test prompt"

    # Verify completion event
    completion_event = mock_event_logger.emit.call_args_list[1][0][0]
    assert completion_event.name == "gen_ai.completion"
    assert completion_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.VERTEXAI.value
    assert completion_event.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gemini-pro"
    assert completion_event.body["content"] == "Test completion"
    assert completion_event.body["safety_attributes"] == {"harmful": 0.1, "hate": 0.0}

def test_event_based_chat(mock_vertexai, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking for chat."""
    # Initialize instrumentation with event-based mode
    instrumentor = VertexAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=Mock(get_tracer=Mock(return_value=mock_tracer)),
        meter_provider=Mock(get_meter=Mock(return_value=mock_meter)),
        event_logger=mock_event_logger,
        use_legacy_attributes=False,
    )

    # Mock chat session and response
    mock_chat = Mock()
    mock_chat.history = [
        {"author": "user", "content": "Hello"},
        {"author": "assistant", "content": "Hi there!"},
    ]
    mock_response = Mock()
    mock_response.text = "How can I help you?"
    mock_vertexai.GenerativeModel.return_value.start_chat.return_value = mock_chat
    mock_chat.send_message.return_value = mock_response

    # Create model and chat
    model = mock_vertexai.GenerativeModel("gemini-pro")
    chat = model.start_chat()
    chat.send_message("Hello")

    # Verify events were logged (2 history messages + new message + response)
    assert mock_event_logger.emit.call_count == 4

    # Verify history message events
    history_event1 = mock_event_logger.emit.call_args_list[0][0][0]
    assert history_event1.name == "gen_ai.prompt"
    assert history_event1.body["role"] == "user"
    assert history_event1.body["content"] == "Hello"

    history_event2 = mock_event_logger.emit.call_args_list[1][0][0]
    assert history_event2.name == "gen_ai.prompt"
    assert history_event2.body["role"] == "assistant"
    assert history_event2.body["content"] == "Hi there!"

    # Verify new message event
    message_event = mock_event_logger.emit.call_args_list[2][0][0]
    assert message_event.name == "gen_ai.prompt"
    assert message_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.VERTEXAI.value
    assert message_event.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gemini-pro"
    assert message_event.body["content"] == "Hello"

    # Verify completion event
    completion_event = mock_event_logger.emit.call_args_list[3][0][0]
    assert completion_event.name == "gen_ai.completion"
    assert completion_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.VERTEXAI.value
    assert completion_event.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gemini-pro"
    assert completion_event.body["content"] == "How can I help you?"

def test_event_based_no_content_capture(mock_vertexai, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking with content capture disabled."""
    # Initialize instrumentation with event-based mode and content capture disabled
    instrumentor = VertexAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=Mock(get_tracer=Mock(return_value=mock_tracer)),
        meter_provider=Mock(get_meter=Mock(return_value=mock_meter)),
        event_logger=mock_event_logger,
        use_legacy_attributes=False,
        capture_content=False,
    )

    # Mock response
    mock_response = Mock()
    mock_response.text = "Test completion"
    mock_response.safety_attributes = {"harmful": 0.1, "hate": 0.0}
    mock_vertexai.GenerativeModel.return_value.generate_content.return_value = mock_response

    # Create model and generate content
    model = mock_vertexai.GenerativeModel("gemini-pro")
    model.generate_content("Test prompt")

    # Verify events were logged
    assert mock_event_logger.emit.call_count == 2

    # Verify prompt event has no content
    prompt_event = mock_event_logger.emit.call_args_list[0][0][0]
    assert prompt_event.name == "gen_ai.prompt"
    assert prompt_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.VERTEXAI.value
    assert prompt_event.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gemini-pro"
    assert "content" not in prompt_event.body

    # Verify completion event has no content
    completion_event = mock_event_logger.emit.call_args_list[1][0][0]
    assert completion_event.name == "gen_ai.completion"
    assert completion_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.VERTEXAI.value
    assert completion_event.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gemini-pro"
    assert "content" not in completion_event.body
    assert "safety_attributes" not in completion_event.body 