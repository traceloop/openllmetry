"""
Test for issue #3395 - OpenAI responses API not emitting spans when using streaming.
This test reproduces the exact scenario from the issue.
"""

import pytest
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr
def test_issue_3395_streaming_spans(instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI):
    """
    Test that reproduces the exact issue from #3395.

    Issue: OpenAI responses API does not emit spans when using stream=True.

    The original code that failed:
    ```python
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": "ping"}],
        stream=True
    )
    for chunk in response:
        print(chunk)
    ```
    """
    # Clear any existing spans
    span_exporter.clear()

    # Create a streaming response - exact pattern from the issue
    response = openai_client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": "ping"}],
        stream=True
    )

    # Consume the stream - exact pattern from the issue
    chunks_received = []
    for chunk in response:
        chunks_received.append(chunk)

    # Verify that spans were created (this was the bug - no spans were created)
    spans = span_exporter.get_finished_spans()

    # MAIN ASSERTION: We should have at least one span (the responses.create span)
    assert len(spans) > 0, "No spans were created for streaming response - issue #3395 not fixed!"

    # Verify the span has the correct attributes
    response_span = None
    for span in spans:
        if span.name == "openai.response":
            response_span = span
            break

    assert response_span is not None, "No 'openai.response' span found"

    # Verify key attributes are present
    assert response_span.attributes.get("gen_ai.system") == "openai"
    assert response_span.attributes.get("gen_ai.request.model") == "gpt-4o-mini"

    # Verify the input was captured
    prompt_content = response_span.attributes.get("gen_ai.prompt.0.content")
    if isinstance(prompt_content, str):
        assert "ping" in prompt_content or prompt_content == "ping"

    # Verify we actually received chunks
    assert len(chunks_received) > 0, "No chunks were received from the stream"


@pytest.mark.vcr
def test_issue_3395_with_context_manager(instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI):
    """
    Test streaming with context manager (another common pattern).
    """
    tracer = trace.get_tracer(__name__)

    # Clear any existing spans
    span_exporter.clear()

    # Similar to the issue example with tracer.start_as_current_span
    with tracer.start_as_current_span("example-span"):
        response = openai_client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": "ping"}],
            stream=True
        )

        chunks_received = []
        for chunk in response:
            chunks_received.append(chunk)

    # Get all spans
    spans = span_exporter.get_finished_spans()

    # We should have both the example-span and the openai.response span
    span_names = [span.name for span in spans]
    assert "openai.response" in span_names, "Missing openai.response span"
    assert "example-span" in span_names, "Missing example-span"

    # Verify we got chunks
    assert len(chunks_received) > 0, "No chunks received"
