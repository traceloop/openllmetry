"""Test event-based tracking for Transformers instrumentation."""

from unittest.mock import Mock, patch
import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.instrumentation.transformers import TransformersInstrumentor

@pytest.fixture
def mock_pipeline():
    with patch("transformers.pipeline") as mock:
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

def test_event_based_generation(mock_pipeline, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking for text generation."""
    # Initialize instrumentation with event-based mode
    instrumentor = TransformersInstrumentor()
    instrumentor.instrument(
        tracer_provider=Mock(get_tracer=Mock(return_value=mock_tracer)),
        meter_provider=Mock(get_meter=Mock(return_value=mock_meter)),
        event_logger=mock_event_logger,
        use_legacy_attributes=False,
    )

    # Mock pipeline response
    mock_pipeline.return_value = Mock()
    mock_pipeline.return_value.return_value = [{"generated_text": "Test completion"}]
    mock_pipeline.return_value.model.name_or_path = "gpt2"

    # Create and use pipeline
    generator = mock_pipeline("text-generation")
    generator("Test prompt")

    # Verify events were logged
    assert mock_event_logger.emit.call_count == 2

    # Verify prompt event
    prompt_event = mock_event_logger.emit.call_args_list[0][0][0]
    assert prompt_event.name == "gen_ai.prompt"
    assert prompt_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.TRANSFORMERS.value
    assert prompt_event.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt2"
    assert prompt_event.body["content"] == "Test prompt"

    # Verify completion event
    completion_event = mock_event_logger.emit.call_args_list[1][0][0]
    assert completion_event.name == "gen_ai.completion"
    assert completion_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.TRANSFORMERS.value
    assert completion_event.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt2"
    assert completion_event.body["content"] == "Test completion"

def test_event_based_generation_no_content_capture(mock_pipeline, mock_tracer, mock_meter, mock_event_logger):
    """Test event-based tracking with content capture disabled."""
    # Initialize instrumentation with event-based mode and content capture disabled
    instrumentor = TransformersInstrumentor()
    instrumentor.instrument(
        tracer_provider=Mock(get_tracer=Mock(return_value=mock_tracer)),
        meter_provider=Mock(get_meter=Mock(return_value=mock_meter)),
        event_logger=mock_event_logger,
        use_legacy_attributes=False,
        capture_content=False,
    )

    # Mock pipeline response
    mock_pipeline.return_value = Mock()
    mock_pipeline.return_value.return_value = [{"generated_text": "Test completion"}]
    mock_pipeline.return_value.model.name_or_path = "gpt2"

    # Create and use pipeline
    generator = mock_pipeline("text-generation")
    generator("Test prompt")

    # Verify events were logged
    assert mock_event_logger.emit.call_count == 2

    # Verify prompt event has no content
    prompt_event = mock_event_logger.emit.call_args_list[0][0][0]
    assert prompt_event.name == "gen_ai.prompt"
    assert prompt_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.TRANSFORMERS.value
    assert prompt_event.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt2"
    assert "content" not in prompt_event.body

    # Verify completion event has no content
    completion_event = mock_event_logger.emit.call_args_list[1][0][0]
    assert completion_event.name == "gen_ai.completion"
    assert completion_event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.TRANSFORMERS.value
    assert completion_event.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt2"
    assert "content" not in completion_event.body 