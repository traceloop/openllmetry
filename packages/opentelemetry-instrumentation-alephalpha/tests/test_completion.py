import os
import pytest
from aleph_alpha_client import Client, Prompt, CompletionRequest
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode


@pytest.mark.vcr
def test_alephalpha_completion_legacy(exporter):
    """Test legacy attribute-based approach."""
    client = Client(token=os.environ.get("AA_TOKEN"))
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
    }
    request = CompletionRequest(**params)
    response = client.complete(request, model="luminous-base")

    spans = exporter.get_finished_spans()
    span = spans[0]
    assert span.name == "alephalpha.completion"
    assert span.attributes.get(SpanAttributes.LLM_SYSTEM) == "AlephAlpha"
    assert span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "completion"
    assert span.attributes.get(SpanAttributes.GEN_AI_REQUEST_MODEL) == "luminous-base"
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry."
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.completions[0].completion
    )
    assert span.attributes.get(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS) == 9
    assert span.attributes.get(
        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS
    ) == span.attributes.get(
        SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS
    ) + span.attributes.get(
        SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS
    )


@pytest.mark.vcr
def test_alephalpha_completion_events(exporter):
    """Test event-based approach."""
    # Configure for event-based approach
    instrumentor = AlephAlphaInstrumentor()
    instrumentor._instrument(use_legacy_attributes=False)

    client = Client(token=os.environ.get("AA_TOKEN"))
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
    }
    request = CompletionRequest(**params)
    response = client.complete(request, model="luminous-base")

    spans = exporter.get_finished_spans()
    span = spans[0]
    
    # Check base attributes
    assert span.name == "alephalpha.completion"
    assert span.attributes.get(SpanAttributes.LLM_SYSTEM) == "AlephAlpha"
    assert span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "completion"
    
    # Check events
    events = span.events
    assert len(events) == 2
    
    prompt_event = events[0]
    assert prompt_event.name == "gen_ai.prompt"
    assert prompt_event.attributes[SpanAttributes.LLM_SYSTEM] == "AlephAlpha"
    assert prompt_event.body["role"] == "user"
    assert prompt_event.body["content"] == prompt_text
    assert prompt_event.trace_id == span.context.trace_id
    assert prompt_event.span_id == span.context.span_id
    
    completion_event = events[1]
    assert completion_event.name == "gen_ai.completion"
    assert completion_event.attributes[SpanAttributes.LLM_SYSTEM] == "AlephAlpha"
    assert completion_event.body["role"] == "assistant"
    assert completion_event.body["content"] == response.completions[0].completion
    assert completion_event.trace_id == span.context.trace_id
    assert completion_event.span_id == span.context.span_id
    
    # Check usage attributes are still present
    assert span.attributes.get(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS) == 9
    assert span.attributes.get(
        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS
    ) == span.attributes.get(
        SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS
    ) + span.attributes.get(
        SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS
    )


@pytest.mark.vcr
def test_alephalpha_completion_streaming(exporter):
    """Test streaming response handling."""
    instrumentor = AlephAlphaInstrumentor()
    instrumentor._instrument(use_legacy_attributes=False)

    client = Client(token=os.environ.get("AA_TOKEN"))
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
        "stream": True,
    }
    request = CompletionRequest(**params)
    
    response_stream = client.complete(request, model="luminous-base")
    completion_text = ""
    for chunk in response_stream:
        if chunk.completions:
            completion_text += chunk.completions[0].completion

    spans = exporter.get_finished_spans()
    span = spans[0]
    
    # Check events
    events = span.events
    assert len(events) == 2
    
    prompt_event = events[0]
    assert prompt_event.name == "gen_ai.prompt"
    assert prompt_event.body["content"] == prompt_text
    
    completion_event = events[1]
    assert completion_event.name == "gen_ai.completion"
    assert completion_event.body["content"] == completion_text


@pytest.mark.vcr
def test_alephalpha_completion_error_handling(exporter):
    """Test error handling in completion requests."""
    instrumentor = AlephAlphaInstrumentor()
    instrumentor._instrument(use_legacy_attributes=False)

    client = Client(token="invalid_token")  # This should cause an auth error
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
    }
    request = CompletionRequest(**params)
    
    with pytest.raises(Exception):  # Should raise an authentication error
        client.complete(request, model="luminous-base")

    spans = exporter.get_finished_spans()
    span = spans[0]
    
    # Check error handling
    assert span.status.status_code == StatusCode.ERROR
    assert len(span.events) == 1  # Only prompt event, no completion due to error
    assert span.events[0].name == "gen_ai.prompt"


@pytest.mark.vcr
def test_alephalpha_completion_streaming_error(exporter):
    """Test error handling in streaming responses."""
    instrumentor = AlephAlphaInstrumentor()
    instrumentor._instrument(use_legacy_attributes=False)

    client = Client(token="invalid_token")
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
        "stream": True,
    }
    request = CompletionRequest(**params)
    
    with pytest.raises(Exception):
        response_stream = client.complete(request, model="luminous-base")
        for _ in response_stream:  # Should raise during iteration
            pass

    spans = exporter.get_finished_spans()
    span = spans[0]
    
    # Check error handling in streaming mode
    assert span.status.status_code == StatusCode.ERROR
    assert len(span.events) == 1  # Only prompt event, stream failed
    assert span.events[0].name == "gen_ai.prompt"
