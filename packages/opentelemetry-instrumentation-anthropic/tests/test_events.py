import pytest
from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.anthropic.config import Config as AnthropicConfig

@pytest.fixture
def reset_config_anthropic():
    """Reset the Config.use_legacy_attributes to its original value after each test for anthropic."""
    original_value = AnthropicConfig.use_legacy_attributes
    yield
    AnthropicConfig.use_legacy_attributes = original_value

# START: Test for Anthropic legacy attributes
def test_anthropic_legacy_attributes(anthropic_exporter, reset_config_anthropic, anthropic_client):
    """Test Anthropic legacy attributes."""

    # Set up legacy mode for Anthropic
    AnthropicConfig.use_legacy_attributes = True

    # Perform a simple completion request
    prompt = "Tell me a joke"
    response = anthropic_client.completions.create(
        model="claude-v1.3", prompt=prompt, max_tokens_to_sample=10
    )

    # Get the span and verify legacy attribute behavior
    spans = anthropic_exporter.get_finished_spans()
    completion_span = spans[0]

    # Check that legacy attributes are present
    assert (
        completion_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.user") == prompt
    )
    assert (
        completion_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.completion
    )

    # Verify that no events are present
    assert not any(event.name == "prompt" for event in completion_span.events)
    assert not any(event.name == "completion" for event in completion_span.events)
# END: Test for Anthropic legacy attributes

# START: Test for Anthropic event-based attributes
def test_anthropic_event_based_attributes(anthropic_exporter, reset_config_anthropic, anthropic_client):
    """Test Anthropic event-based attributes."""

    # Set up event-based mode for Anthropic
    AnthropicConfig.use_legacy_attributes = False

    # Perform a simple completion request
    prompt = "Tell me a joke"
    response = anthropic_client.completions.create(
        model="claude-v1.3", prompt=prompt, max_tokens_to_sample=10
    )

    # Get the span and verify event-based behavior
    spans = anthropic_exporter.get_finished_spans()
    completion_span = spans[0]

    # Check that legacy attributes are not present
    assert completion_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.user") is None
    assert completion_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") is None

    # Verify that events are present
    prompt_events = [event for event in completion_span.events if event.name == "prompt"]
    completion_events = [event for event in completion_span.events if event.name == "completion"]

    assert len(prompt_events) == 1
    assert prompt_events[0].attributes["messaging.role"] == "user"
    assert prompt_events[0].attributes["messaging.content"] == prompt
    assert prompt_events[0].attributes["messaging.index"] == 0

    assert len(completion_events) == 1
    assert completion_events[0].attributes["messaging.content"] == response.completion
    assert completion_events[0].attributes["messaging.index"] == 0

    if hasattr(response, "usage"):
        assert completion_events[0].attributes["llm.usage.total_tokens"] == response.usage.total_tokens
        assert completion_events[0].attributes["llm.usage.prompt_tokens"] == response.usage.prompt_tokens
        assert completion_events[0].attributes["llm.usage.completion_tokens"] == response.usage.completion_tokens
# END: Test for Anthropic event-based attributes