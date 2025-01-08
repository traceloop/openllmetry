import os
import pytest
from opentelemetry.semconv_ai import SpanAttributes
import aleph_alpha_client
from opentelemetry.instrumentation.alephalpha.config import Config

@pytest.fixture
def reset_config():
    """Reset the Config.use_legacy_attributes to its original value after each test."""
    original_value = Config().use_legacy_attributes
    Config.use_legacy_attributes = original_value
    yield
    Config().use_legacy_attributes = original_value

def _create_client():
    api_key = os.environ.get("ALEPH_ALPHA_API_KEY")
    if not api_key:
        pytest.skip("ALEPH_ALPHA_API_KEY environment variable not set.")
    return aleph_alpha_client.Client(
        token=api_key,
        host=os.environ.get("ALEPH_ALPHA_API_HOST", "https://api.aleph-alpha.com")
    )

def test_legacy_attributes(exporter, reset_config):
    """Test that legacy attributes are correctly set when use_legacy_attributes is True."""
    # Set up legacy mode
    Config().use_legacy_attributes = True
    client = _create_client()

    # Perform a simple completion request
    prompt = "Tell me a joke"
    response = client.complete(
        prompt=aleph_alpha_client.Prompt.from_text(prompt), model="luminous-base"
    )

    # Get the span and verify legacy attribute behavior
    spans = exporter.get_finished_spans()
    completion_span = spans[0]

    # Check that legacy attributes are present
    assert completion_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content") == prompt
    assert completion_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.completions[0].completion

    # Verify that no events are present (since we're in legacy mode)
    assert not any(event.name == "prompt" for event in completion_span.events)
    assert not any(event.name == "completion" for event in completion_span.events)

def test_event_based_attributes(exporter, reset_config):
    """Test that events are correctly emitted when use_legacy_attributes is False."""
    # Set up event-based mode
    Config().use_legacy_attributes = False
    client = _create_client()

    # Perform a simple completion request
    prompt = "Tell me a joke"
    response = client.complete(
        prompt=aleph_alpha_client.Prompt.from_text(prompt), model="luminous-base"
    )

    # Get the span and verify event-based behavior
    spans = exporter.get_finished_spans()
    completion_span = spans[0]

    # Check that legacy attributes are not present
    assert completion_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content") is None
    assert completion_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") is None

    # Verify that events are present with correct attributes
    prompt_events = [event for event in completion_span.events if event.name == "prompt"]
    completion_events = [event for event in completion_span.events if event.name == "completion"]

    # Check prompt event
    assert len(prompt_events) == 1
    assert prompt_events[0].attributes["messaging.role"] == "user"
    assert prompt_events[0].attributes["messaging.content"] == prompt
    assert prompt_events[0].attributes["messaging.index"] == 0

    # Check completion event
    assert len(completion_events) == 1
    assert completion_events[0].attributes["messaging.content"] == response.completions[0].completion
    assert completion_events[0].attributes["messaging.role"] == "assistant"
    assert completion_events[0].attributes["messaging.index"] == 0

    # Check token usage in completion event
    assert completion_span.attributes["llm.usage.total_tokens"] == response.num_tokens_prompt_total + response.num_tokens_generated
    assert completion_span.attributes["llm.usage.prompt_tokens"] == response.num_tokens_prompt_total
    assert completion_span.attributes["llm.usage.completion_tokens"] == response.num_tokens_generated

def _create_client():
    api_key = os.environ.get("ALEPH_ALPHA_API_KEY")
    if not api_key:
        pytest.skip("ALEPH_ALPHA_API_KEY environment variable not set.")
    return aleph_alpha_client.Client(
        token=api_key,
        host=os.environ.get("ALEPH_ALPHA_API_HOST", "https://api.aleph-alpha.com")
    )