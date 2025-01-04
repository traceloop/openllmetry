import os
import pytest
from opentelemetry.semconv_ai import SpanAttributes
import cohere
from opentelemetry.instrumentation.cohere.config import Config

@pytest.fixture
def reset_config():
    """Reset the Config.use_legacy_attributes to its original value after each test."""
    original_value = Config.use_legacy_attributes
    yield
    Config.use_legacy_attributes = original_value

def test_legacy_attributes(exporter, reset_config):
    """Test that legacy attributes are correctly set when use_legacy_attributes is True."""
    # Set up legacy mode
    Config.use_legacy_attributes = True
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))

    # Perform a simple chat request
    message = "Tell me a joke"
    response = co.chat(model="command", message=message)

    # Get the span and verify legacy attribute behavior
    spans = exporter.get_finished_spans()
    chat_span = spans[0]

    # Check that legacy attributes are present
    assert chat_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content") == message
    assert chat_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.text

    # Verify that no events are present (since we're in legacy mode)
    assert not any(event.name == "prompt" for event in chat_span.events)
    assert not any(event.name == "completion" for event in chat_span.events)

def test_event_based_attributes(exporter, reset_config):
    """Test that events are correctly emitted when use_legacy_attributes is False."""
    # Set up event-based mode
    Config.use_legacy_attributes = False
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))

    # Perform a simple chat request
    message = "Tell me a joke"
    response = co.chat(model="command", message=message)

    # Get the span and verify event-based behavior
    spans = exporter.get_finished_spans()
    chat_span = spans[0]

    # Check that legacy attributes are not present
    assert chat_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content") is None
    assert chat_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") is None

    # Verify that events are present with correct attributes
    prompt_events = [event for event in chat_span.events if event.name == "prompt"]
    completion_events = [event for event in chat_span.events if event.name == "completion"]

    # Check prompt event
    assert len(prompt_events) == 1
    assert prompt_events[0].attributes["messaging.role"] == "user"
    assert prompt_events[0].attributes["messaging.content"] == message
    assert prompt_events[0].attributes["messaging.index"] == 0

    # Check completion event
    assert len(completion_events) == 1
    assert completion_events[0].attributes["messaging.content"] == response.text
    assert completion_events[0].attributes["messaging.index"] == 0

    # Check token usage in completion event (if available)
    if hasattr(response, "token_count"):
        assert completion_events[0].attributes["llm.usage.total_tokens"] == response.token_count.get("total_tokens")
        assert completion_events[0].attributes["llm.usage.prompt_tokens"] == response.token_count.get("prompt_tokens")
        assert completion_events[0].attributes["llm.usage.completion_tokens"] == response.token_count.get("response_tokens")