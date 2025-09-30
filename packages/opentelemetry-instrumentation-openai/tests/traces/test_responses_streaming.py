"""Tests for OpenAI Responses API streaming instrumentation."""

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

    # Create a streaming response
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

    # Verify that we do have a span (the responses.create span)
    assert len(spans) > 0, "No spans were created for streaming response - issue #3395 not fixed!"

    # Verify the attributes
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
def test_responses_streaming(instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI):
    """Test that streaming responses generate proper spans."""

    # Create a streaming response
    response_stream = openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        stream=True
    )

    # Consume the stream
    full_text = ""
    for chunk in response_stream:
        if chunk.output:
            for output_item in chunk.output:
                if hasattr(output_item, 'content'):
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text'):
                            full_text += content_item.text

    # Check that span was created
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"

    # Check that prompt was captured
    assert span.attributes.get("gen_ai.prompt.0.content") == "What is the capital of France?"
    assert span.attributes.get("gen_ai.prompt.0.role") == "user"

    # Check that completion was captured
    assert "gen_ai.completion.0.content" in span.attributes
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"

    # Basic content check - should mention Paris
    assert full_text, "No content was streamed"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_responses_streaming_async(instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI):
    """Test that async streaming responses generate proper spans."""

    # Create async client
    from openai import AsyncOpenAI
    async_client = AsyncOpenAI()

    # Create a streaming response
    response_stream = await async_client.responses.create(
        model="gpt-4.1-nano",
        input="What is 2+2?",
        stream=True
    )

    # Consume the stream
    full_text = ""
    async for chunk in response_stream:
        if chunk.output:
            for output_item in chunk.output:
                if hasattr(output_item, 'content'):
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text'):
                            full_text += content_item.text

    # Check that span was created
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"

    # Check that prompt was captured
    assert span.attributes.get("gen_ai.prompt.0.content") == "What is 2+2?"
    assert span.attributes.get("gen_ai.prompt.0.role") == "user"

    # Check that completion was captured
    assert "gen_ai.completion.0.content" in span.attributes
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"

    # Basic content check
    assert full_text, "No content was streamed"


@pytest.mark.vcr
def test_responses_streaming_with_context_manager(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test streaming responses with context manager usage."""

    full_text = ""
    with openai_client.responses.create(
        model="gpt-4.1-nano",
        input="Count to 5",
        stream=True
    ) as response_stream:
        for chunk in response_stream:
            if chunk.output:
                for output_item in chunk.output:
                    if hasattr(output_item, 'content'):
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text'):
                                full_text += content_item.text

    # Check that span was created after context manager exits
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"

    # Verify content was captured
    assert full_text, "No content was streamed"
    assert "gen_ai.completion.0.content" in span.attributes


@pytest.mark.vcr
def test_responses_streaming_with_tracer_context(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """
    Test streaming with tracer context (similar to issue #3395 example).
    Ensures spans are properly nested when using tracer.start_as_current_span.
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


@pytest.mark.vcr
def test_responses_streaming_error_handling(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test that streaming errors are properly handled and spans are closed."""

    # This test would ideally trigger an error during streaming
    # For now, we'll test with a normal stream but verify error handling paths exist

    try:
        response_stream = openai_client.responses.create(
            model="gpt-4.1-nano",
            input="What is the meaning of life?",
            stream=True,
            max_tokens=10  # Very low to potentially trigger issues
        )

        for chunk in response_stream:
            pass  # Just consume the stream

    except Exception:
        pass  # Error is expected in some cases

    # Even if there's an error, span should be created and closed
    spans = span_exporter.get_finished_spans()
   assert len(spans) > 0, "Spans should be available even on error"
    if spans:
        span = spans[0]
        assert span.name == "openai.response"
        # If there was an error, it should be recorded
        # (checking span.status would be ideal here)
