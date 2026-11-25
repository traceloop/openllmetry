import json
import threading
import time
import types
from collections.abc import AsyncIterator, Iterator

import pytest
from pydantic import BaseModel

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.responses import ResponseStreamEvent
from opentelemetry.instrumentation.openai.utils import is_reasoning_supported
from opentelemetry.instrumentation.openai.v1 import responses_wrappers
from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
    ResponseStream,
    async_responses_get_or_create_wrapper,
    get_tools_from_kwargs,
    responses_cancel_wrapper,
    responses_get_or_create_wrapper,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from .utils import get_input_messages, get_output_messages


class Person(BaseModel):
    name: str
    age: int


@pytest.mark.vcr
def test_responses(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    _ = openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"


@pytest.mark.vcr
def test_responses_with_request_params(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test that request parameters like temperature, max_tokens, top_p are captured"""
    _ = openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        temperature=0.7,
        max_output_tokens=100,
        top_p=0.9,
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"

    # Check that request parameters are captured
    assert span.attributes["gen_ai.request.temperature"] == 0.7
    assert span.attributes["gen_ai.request.max_tokens"] == 100
    assert span.attributes["gen_ai.request.top_p"] == 0.9


@pytest.mark.vcr
def test_responses_with_service_tier(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    _ = openai_client.responses.create(
        model="gpt-5",
        input="Say hello",
        service_tier="priority",
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["openai.request.service_tier"] == "priority"
    assert span.attributes["openai.response.service_tier"] == "priority"


@pytest.mark.vcr
def test_responses_with_input_history(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    user_message = "Come up with an adjective in English. Respond with just one word."
    first_response = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=user_message,
    )
    _ = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": first_response.output[0].content[0].text,
                    }
                ],
            },
            {"role": "user", "content": "Can you explain why you chose that word?"},
        ],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[1]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"


@pytest.mark.vcr
def test_responses_tool_calls(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]
    openai_client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in London?",
            }
        ],
        tools=tools,
        tool_choice="auto",
    )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"

    assert (
        span.attributes["gen_ai.response.id"]
        == "resp_685ff8928dc4819aac45e085ba66838101c537ddeff5c2a2"
    )


@pytest.mark.vcr
@pytest.mark.skipif(
    not is_reasoning_supported(),
    reason="Reasoning is not supported in older OpenAI library versions",
)
def test_responses_reasoning(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    openai_client.responses.create(
        model="gpt-5-nano",
        input="Count r's in strawberry",
        reasoning={"effort": "low", "summary": None},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1


@pytest.mark.vcr
@pytest.mark.skipif(
    not is_reasoning_supported(),
    reason="Reasoning is not supported in older OpenAI library versions",
)
def test_responses_reasoning_dict_issue(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test for issue #3350 - reasoning dict causing invalid type warning"""
    openai_client.responses.create(
        model="gpt-5-nano",
        input="Explain why the sky is blue",
        reasoning={"effort": "medium", "summary": "auto"},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Verify the reasoning attributes are properly set without causing warnings
    # The main goal of this test is to ensure that when the API returns reasoning data
    # as a dict/list, it gets properly serialized as JSON without causing "Invalid type" warnings

    # Reasoning content is embedded in gen_ai.output.messages as a reasoning part
    output_messages = get_output_messages(span)
    assert len(output_messages) > 0, "Expected at least one output message"

    # Find any reasoning parts across all output messages
    reasoning_parts = [
        p for msg in output_messages
        for p in msg.get("parts", [])
        if p.get("type") == "reasoning"
    ]

    # If reasoning parts exist, verify their content is a properly serialized string
    for part in reasoning_parts:
        content = part.get("content")
        assert isinstance(content, str), (
            f"Reasoning content should be a string (not raw dict/list), got: {type(content)}"
        )
        # If it looks like JSON, verify it parses correctly
        if content and content.strip().startswith(("[", "{")):
            parsed = json.loads(content)
            assert isinstance(parsed, (dict, list)), (
                "Reasoning content that looks like JSON should parse to dict or list"
            )


@pytest.mark.vcr
def test_responses_streaming(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test for streaming responses.create() - reproduces customer issue"""
    input_text = "Tell me a three sentence bedtime story about a unicorn."
    stream = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=input_text,
        stream=True,
    )

    # Consume the stream
    full_text = ""
    for item in stream:
        if hasattr(item, "type") and item.type == "response.output_text.delta":
            if hasattr(item, "delta") and item.delta:
                full_text += item.delta
        elif hasattr(item, "delta") and item.delta:
            if hasattr(item.delta, "text") and item.delta.text:
                full_text += item.delta.text

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span but got {len(spans)}"

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert full_text != "", "Should have received streaming content"
    input_messages = get_input_messages(span)
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == input_text
    output_messages = get_output_messages(span)
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["content"] == full_text


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_responses_streaming_async(
    instrument_legacy, span_exporter: InMemorySpanExporter, async_openai_client
):
    """Test for async streaming responses.create() - reproduces customer issue"""
    input_text = "Tell me a three sentence bedtime story about a unicorn."
    stream = await async_openai_client.responses.create(
        model="gpt-4.1-nano",
        input=input_text,
        stream=True,
    )

    full_text = ""
    async for item in stream:
        if hasattr(item, "type") and item.type == "response.output_text.delta":
            if hasattr(item, "delta") and item.delta:
                full_text += item.delta
        elif hasattr(item, "delta") and item.delta:
            if hasattr(item.delta, "text") and item.delta.text:
                full_text += item.delta.text

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span but got {len(spans)}"

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"
    input_messages = get_input_messages(span)
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == input_text
    output_messages = get_output_messages(span)
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["content"] == full_text


@pytest.mark.vcr
def test_responses_streaming_with_content(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test streaming with content tracing - verifies prompts and completions are captured"""
    input_text = "What is 2+2?"
    stream = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=input_text,
        stream=True,
    )

    # Consume the stream
    full_text = ""
    for item in stream:
        if hasattr(item, "type") and item.type == "response.output_text.delta":
            if hasattr(item, "delta") and item.delta:
                full_text += item.delta

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"
    input_messages = get_input_messages(span)
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == input_text
    output_messages = get_output_messages(span)
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["content"] == full_text


@pytest.mark.vcr
def test_responses_streaming_with_context_manager(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test streaming responses using context manager (with statement)"""
    input_text = "Count to 5"
    full_text = ""

    with openai_client.responses.create(
        model="gpt-4.1-nano",
        input=input_text,
        stream=True,
    ) as stream:
        for chunk in stream:
            if chunk.type == "response.output_text.delta":
                full_text += chunk.delta

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"
    input_messages = get_input_messages(span)
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == input_text
    output_messages = get_output_messages(span)
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["content"] == full_text


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_responses_streaming_async_with_context_manager(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    """Test async streaming responses using context manager (async with statement)"""
    input_text = "Count to 5"
    full_text = ""

    stream = await async_openai_client.responses.create(
        model="gpt-4.1-nano",
        input=input_text,
        stream=True,
    )

    async with stream:
        async for chunk in stream:
            if chunk.type == "response.output_text.delta":
                full_text += chunk.delta

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"
    input_messages = get_input_messages(span)
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == input_text
    output_messages = get_output_messages(span)
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["content"] == full_text


@pytest.mark.vcr
def test_responses_parse(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Structured-output via responses.parse() must produce an LLM span with usage
    and capture the prompt + structured response on the span."""
    input_text = "Extract: Alice is 30 years old."
    response = openai_client.responses.parse(
        model="gpt-4.1-nano",
        input=input_text,
        text_format=Person,
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"expected one openai.response span, got {len(spans)}"
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.usage.input_tokens"] > 0
    assert span.attributes["gen_ai.usage.output_tokens"] > 0

    input_messages = get_input_messages(span)
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == input_text

    output_messages = get_output_messages(span)
    assert output_messages[0]["role"] == "assistant"
    output_text = output_messages[0]["parts"][0]["content"]
    parsed = Person.model_validate_json(output_text)
    assert parsed == response.output_parsed
    assert parsed.name == "Alice"
    assert parsed.age == 30


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_responses_parse_async(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    """Async structured-output via responses.parse() must produce an LLM span with usage
    and capture the prompt + structured response on the span."""
    input_text = "Extract: Bob is 42 years old."
    response = await async_openai_client.responses.parse(
        model="gpt-4.1-nano",
        input=input_text,
        text_format=Person,
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"expected one openai.response span, got {len(spans)}"
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.usage.input_tokens"] > 0
    assert span.attributes["gen_ai.usage.output_tokens"] > 0

    input_messages = get_input_messages(span)
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == input_text

    output_messages = get_output_messages(span)
    assert output_messages[0]["role"] == "assistant"
    output_text = output_messages[0]["parts"][0]["content"]
    parsed = Person.model_validate_json(output_text)
    assert parsed == response.output_parsed
    assert parsed.name == "Bob"
    assert parsed.age == 42


def test_get_tools_from_kwargs_with_none():
    """Test that get_tools_from_kwargs handles None tools value correctly.

    This reproduces the bug reported when openai-guardrails or other wrappers
    pass tools=None explicitly, causing TypeError: 'NoneType' object is not iterable.
    """
    # Test case 1: tools key exists but value is None
    kwargs_with_none = {"tools": None, "model": "gpt-4", "input": "test"}
    result = get_tools_from_kwargs(kwargs_with_none)
    assert result == [], "Should return empty list when tools is None"

    # Test case 2: tools key doesn't exist
    kwargs_without_tools = {"model": "gpt-4", "input": "test"}
    result = get_tools_from_kwargs(kwargs_without_tools)
    assert result == [], "Should return empty list when tools key is missing"

    # Test case 3: tools is an empty list
    kwargs_empty_list = {"tools": [], "model": "gpt-4", "input": "test"}
    result = get_tools_from_kwargs(kwargs_empty_list)
    assert result == [], "Should return empty list when tools is empty list"

    # Test case 4: tools with valid function tools
    kwargs_with_tools = {
        "tools": [{"type": "function", "name": "test_func", "description": "test"}],
        "model": "gpt-4",
        "input": "test",
    }
    result = get_tools_from_kwargs(kwargs_with_tools)
    assert len(result) == 1, "Should return list with one tool"


def test_response_stream_init_with_none_tools():
    """Test ResponseStream initialization when tools=None is in request_kwargs.

    This reproduces the customer issue where openai-guardrails wraps the client
    and may pass tools=None, causing TypeError in ResponseStream.__init__.
    """
    from unittest.mock import MagicMock
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
        ResponseStream,
    )

    # Create a mock response object
    mock_response = MagicMock()

    # Create a mock span
    mock_span = MagicMock()

    # Create a mock tracer
    mock_tracer = MagicMock()

    # Test that ResponseStream can be initialized with tools=None
    # This should not raise TypeError: 'NoneType' object is not iterable
    request_kwargs_with_none_tools = {
        "model": "gpt-4",
        "input": "test",
        "tools": None,  # This is what causes the bug
        "stream": True,
    }

    # This should not raise an exception
    stream = ResponseStream(
        span=mock_span,
        response=mock_response,
        start_time=1234567890,
        request_kwargs=request_kwargs_with_none_tools,
        tracer=mock_tracer,
    )

    # Verify the stream was created successfully
    assert stream is not None
    assert stream._traced_data is not None
    # Tools should be an empty list, not None
    assert stream._traced_data.tools == [] or stream._traced_data.tools is None


def test_responses_trace_context_propagation_unit():
    """Unit test for trace context propagation in responses API.

    This test verifies that when TracedData is created with a trace context,
    and later a span is created from that TracedData, the span uses the correct
    trace context that was captured at creation time.

    This is critical for guardrails and other wrappers that make multiple calls
    across different execution contexts.

    Note: This is a unit test that simulates what guardrails does. For integration
    testing with the actual openai-guardrails library, see the sample app at:
    packages/sample-app/sample_app/openai_guardrails_example.py
    """
    from opentelemetry import trace, context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import TracedData
    import time

    # Set up tracing
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)

    # Create a parent span and capture its trace context
    with tracer.start_as_current_span("parent-span") as parent_span:
        parent_trace_id = parent_span.get_span_context().trace_id
        parent_context = context.get_current()

        # Create TracedData with the current trace context (simulating responses.create)
        traced_data = TracedData(
            start_time=time.time_ns(),
            response_id="test-response-id",
            input="What is 2+2?",
            instructions=None,
            tools=None,
            output_blocks={},
            usage=None,
            output_text="4",
            request_model="gpt-4.1-nano",
            response_model="gpt-4.1-nano-2025-04-14",
            trace_context=parent_context,
        )

    # Now we're outside the parent span context
    # Simulate creating a span with the stored trace context (like responses.retrieve does)
    ctx = traced_data.trace_context
    span = tracer.start_span(
        "openai.response",
        context=ctx,
        start_time=traced_data.start_time,
    )
    span.end()

    # Verify the span has the correct trace context
    spans = exporter.get_finished_spans()
    parent_spans = [s for s in spans if s.name == "parent-span"]
    openai_spans = [s for s in spans if s.name == "openai.response"]

    assert len(parent_spans) == 1
    assert len(openai_spans) == 1

    # The openai.response span should have the same trace_id as the parent
    assert openai_spans[0].context.trace_id == parent_trace_id, (
        f"openai.response span trace_id ({openai_spans[0].context.trace_id}) "
        f"should match parent trace_id ({parent_trace_id})"
    )

    # The openai.response span should be a child of the parent span
    assert (
        openai_spans[0].parent.span_id == parent_spans[0].context.span_id
    ), "openai.response span should be a child of parent-span"


@pytest.mark.vcr
def test_responses_streaming_with_parent_span(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    tracer_provider,
    openai_client: OpenAI,
):
    """Integration test for trace context propagation with sync streaming responses.

    This test simulates what openai-guardrails does: wrapping OpenAI calls
    with a parent span. It verifies that:
    1. The streaming response span maintains the parent trace context
    2. All spans share the same trace_id
    3. The response span is properly nested as a child of the parent span

    This prevents regressions of the issue where streaming responses would
    create separate traces instead of maintaining trace continuity.
    """
    # Get tracer from the provider used by the test fixtures
    tracer = tracer_provider.get_tracer(__name__)

    # Create a parent span (simulating what guardrails wrapper does)
    with tracer.start_as_current_span("guardrails-wrapper") as parent_span:
        parent_trace_id = parent_span.get_span_context().trace_id
        parent_span_id = parent_span.get_span_context().span_id

        # Make a sync streaming responses.create() call
        # This should create a child span under the parent
        stream = openai_client.responses.create(
            model="gpt-4o",
            input="Count to 3",
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            if chunk.type == "response.output_text.delta":
                full_text += chunk.delta

    # Verify span hierarchy
    spans = span_exporter.get_finished_spans()
    parent_spans = [s for s in spans if s.name == "guardrails-wrapper"]
    openai_spans = [s for s in spans if s.name == "openai.response"]

    assert len(parent_spans) == 1, "Should have exactly one parent span"
    assert len(openai_spans) == 1, "Should have exactly one OpenAI response span"

    openai_span = openai_spans[0]

    # Verify the openai.response span has the same trace_id as the parent
    assert openai_span.context.trace_id == parent_trace_id, (
        f"OpenAI span trace_id ({hex(openai_span.context.trace_id)}) "
        f"should match parent trace_id ({hex(parent_trace_id)}). "
        "If they differ, trace context is not being propagated correctly."
    )

    # Verify the openai.response span is a child of the parent span
    assert openai_span.parent is not None, "OpenAI span should have a parent"
    assert openai_span.parent.span_id == parent_span_id, (
        f"OpenAI span parent_id ({hex(openai_span.parent.span_id)}) "
        f"should match parent span_id ({hex(parent_span_id)}). "
        "The span should be properly nested under the parent."
    )

    # Verify streaming worked correctly
    assert full_text != "", "Should have received streaming content"
    assert openai_span.attributes["gen_ai.provider.name"] == "openai"
    assert openai_span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert openai_span.attributes["gen_ai.is_streaming"] is True


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_responses_streaming_async_with_parent_span(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    tracer_provider,
    async_openai_client: AsyncOpenAI,
):
    """Integration test for trace context propagation with async streaming responses.

    This test simulates what openai-guardrails does: wrapping OpenAI calls
    with a parent span. It verifies that:
    1. The streaming response span maintains the parent trace context
    2. All spans share the same trace_id
    3. The response span is properly nested as a child of the parent span

    This prevents regressions of the issue where streaming responses would
    create separate traces instead of maintaining trace continuity.
    """
    # Get tracer from the provider used by the test fixtures
    tracer = tracer_provider.get_tracer(__name__)

    # Create a parent span (simulating what guardrails wrapper does)
    with tracer.start_as_current_span("guardrails-wrapper") as parent_span:
        parent_trace_id = parent_span.get_span_context().trace_id
        parent_span_id = parent_span.get_span_context().span_id

        # Make an async streaming responses.create() call
        # This should create a child span under the parent
        stream = await async_openai_client.responses.create(
            model="gpt-4o",
            input="Count to 3",
            stream=True,
        )

        full_text = ""
        async for chunk in stream:
            if chunk.type == "response.output_text.delta":
                full_text += chunk.delta

    # Verify span hierarchy
    spans = span_exporter.get_finished_spans()
    parent_spans = [s for s in spans if s.name == "guardrails-wrapper"]
    openai_spans = [s for s in spans if s.name == "openai.response"]

    assert len(parent_spans) == 1, "Should have exactly one parent span"
    assert len(openai_spans) == 1, "Should have exactly one OpenAI response span"

    openai_span = openai_spans[0]

    # Verify the openai.response span has the same trace_id as the parent
    assert openai_span.context.trace_id == parent_trace_id, (
        f"OpenAI span trace_id ({hex(openai_span.context.trace_id)}) "
        f"should match parent trace_id ({hex(parent_trace_id)}). "
        "If they differ, trace context is not being propagated correctly."
    )

    # Verify the openai.response span is a child of the parent span
    assert openai_span.parent is not None, "OpenAI span should have a parent"
    assert openai_span.parent.span_id == parent_span_id, (
        f"OpenAI span parent_id ({hex(openai_span.parent.span_id)}) "
        f"should match parent span_id ({hex(parent_span_id)}). "
        "The span should be properly nested under the parent."
    )

    # Verify streaming worked correctly
    assert full_text != "", "Should have received streaming content"
    assert openai_span.attributes["gen_ai.provider.name"] == "openai"
    assert openai_span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert openai_span.attributes["gen_ai.is_streaming"] is True


def test_response_stream_init_with_not_given_reasoning():
    """Test ResponseStream initialization when reasoning=NOT_GIVEN sentinel.

    This reproduces issue #3472 - OpenAI SDK uses NOT_GIVEN/Omit sentinels for
    unset optional parameters. When code chains .get() calls like
    kwargs.get("reasoning", {}).get(...), it fails because the sentinel exists
    as the key value (not the default {}) but lacks a .get() method.
    """
    from unittest.mock import MagicMock

    try:
        from openai._types import NOT_GIVEN
    except ImportError:
        pytest.skip("NOT_GIVEN sentinel not available in this OpenAI SDK version")

    from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
        ResponseStream,
    )

    mock_response = MagicMock()
    mock_span = MagicMock()
    mock_tracer = MagicMock()

    # Simulate kwargs where reasoning is set to NOT_GIVEN sentinel
    # This is what happens when client.responses.create() is called without
    # explicitly setting the reasoning parameter
    request_kwargs_with_not_given = {
        "model": "gpt-4",
        "input": "test",
        "reasoning": NOT_GIVEN,  # This causes AttributeError: 'NotGiven' has no 'get'
        "stream": True,
    }

    # This should not raise AttributeError
    stream = ResponseStream(
        span=mock_span,
        response=mock_response,
        start_time=1234567890,
        request_kwargs=request_kwargs_with_not_given,
        tracer=mock_tracer,
    )

    assert stream is not None
    assert stream._traced_data is not None
    # Reasoning summary should be None when NOT_GIVEN sentinel is passed
    assert stream._traced_data.request_reasoning_summary is None


def test_response_stream_init_with_omit_reasoning():
    """Test ResponseStream initialization when reasoning=Omit() instance.

    This is a variant of issue #3472 testing the Omit sentinel class.
    """
    from unittest.mock import MagicMock

    try:
        from openai._types import Omit
    except ImportError:
        pytest.skip("Omit sentinel not available in this OpenAI SDK version")

    from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
        ResponseStream,
    )

    mock_response = MagicMock()
    mock_span = MagicMock()
    mock_tracer = MagicMock()

    request_kwargs_with_omit = {
        "model": "gpt-4",
        "input": "test",
        "reasoning": Omit(),  # Another sentinel type that lacks .get()
        "stream": True,
    }

    # This should not raise AttributeError
    stream = ResponseStream(
        span=mock_span,
        response=mock_response,
        start_time=1234567890,
        request_kwargs=request_kwargs_with_omit,
        tracer=mock_tracer,
    )

    assert stream is not None
    assert stream._traced_data is not None
    assert stream._traced_data.request_reasoning_summary is None


def test_parse_response_unwraps_legacy_api_response():
    """Regression test for https://github.com/traceloop/openllmetry/issues/4058:
    parse_response must unwrap LegacyAPIResponse (what with_raw_response currently
    returns for the Responses API) so downstream code can access .id without crashing."""
    from unittest.mock import MagicMock
    from openai._legacy_response import LegacyAPIResponse
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import parse_response

    inner = MagicMock()
    inner.id = "resp_123"

    wrapper = MagicMock(spec=LegacyAPIResponse)
    wrapper.parse.return_value = inner

    result = parse_response(wrapper)

    wrapper.parse.assert_called_once()
    assert result is inner
    assert result.id == "resp_123"


def test_parse_response_unwraps_api_response():
    """parse_response must also unwrap APIResponse/AsyncAPIResponse in case
    the OpenAI SDK changes with_raw_response to return the non-legacy variants."""
    from unittest.mock import MagicMock
    from openai._response import APIResponse
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import parse_response

    inner = MagicMock()
    inner.id = "resp_456"

    wrapper = MagicMock(spec=APIResponse)
    wrapper.parse.return_value = inner

    result = parse_response(wrapper)

    wrapper.parse.assert_called_once()
    assert result is inner
    assert result.id == "resp_456"


@pytest.mark.asyncio
async def test_async_parse_response_unwraps_async_api_response():
    """async_parse_response must unwrap AsyncAPIResponse using await."""
    from unittest.mock import AsyncMock, MagicMock
    from openai._response import AsyncAPIResponse
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import async_parse_response

    inner = MagicMock()
    inner.id = "resp_789"

    wrapper = MagicMock(spec=AsyncAPIResponse)
    wrapper.parse = AsyncMock(return_value=inner)

    result = await async_parse_response(wrapper)

    wrapper.parse.assert_awaited_once()
    assert result is inner
    assert result.id == "resp_789"


def test_parse_response_passes_through_plain_response():
    """parse_response should return a plain Response object unchanged."""
    from unittest.mock import MagicMock
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import parse_response

    plain = MagicMock()
    plain.id = "resp_456"

    result = parse_response(plain)

    assert result is plain


# with_raw_response sends the same request as create(), so the raw-response tests replay the
# recordings of test_responses_streaming and test_responses_streaming_async instead of a hand-made body.
_STORY_PROMPT = "Tell me a three sentence bedtime story about a unicorn."
_FORWARDED_EVENT_TYPES = ("response.output_text.delta", "response.completed")


class CustomResponseStream(Stream[ResponseStreamEvent]):
    def __iter__(self) -> Iterator[ResponseStreamEvent]:
        return (
            event
            for event in super().__iter__()
            if event.type in _FORWARDED_EVENT_TYPES
        )


class CustomAsyncResponseStream(AsyncStream[ResponseStreamEvent]):
    async def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        async for event in super().__aiter__():
            if event.type in _FORWARDED_EVENT_TYPES:
                yield event


def _streamed_text(events: list[ResponseStreamEvent]) -> str:
    return "".join(
        event.delta for event in events if event.type == "response.output_text.delta"
    )


def _assert_event_types(
    events: list[ResponseStreamEvent], stream_type: type[Stream] | type[AsyncStream] | None
) -> None:
    """A custom stream class keeps its own __iter__, which drops the lifecycle events."""
    event_types = {event.type for event in events}
    if stream_type is None:
        assert "response.created" in event_types
    else:
        assert event_types == set(_FORWARDED_EVENT_TYPES)


def _assert_streamed_span(
    span_exporter: InMemorySpanExporter, events: list[ResponseStreamEvent]
) -> ReadableSpan:
    """Assert the one openai.response span holds the prompt and the streamed text, and return it."""
    full_text = _streamed_text(events)
    assert full_text != "", "Should have received streaming content"
    (span,) = [
        span for span in span_exporter.get_finished_spans() if span.name == "openai.response"
    ]
    assert span.status.status_code == StatusCode.OK
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert get_input_messages(span)[0]["parts"][0]["content"] == _STORY_PROMPT
    assert get_output_messages(span)[0]["parts"][0]["content"] == full_text
    return span


def _assert_tracing_failure_span(
    span_exporter: InMemorySpanExporter, events: list[ResponseStreamEvent]
) -> None:
    """The stream still works and the only span ended with the tracing error."""
    assert _streamed_text(events) != ""
    (span,) = span_exporter.get_finished_spans()
    assert span.name == "openai.response"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "proxy construction failed"
    assert span.attributes["error.type"] == "RuntimeError"


def _fail_stream_proxy(**kwargs: object) -> None:
    raise RuntimeError("proxy construction failed")


def test_create_response_stream_sampler_failure_returns_untraced_stream() -> None:
    """Regression test: tracer.start_span() used to run before the try block in
    _create_response_stream, so a failure there (e.g. a broken sampler) reached the caller
    directly instead of falling back to the untraced stream the way every other setup
    failure in this function already does."""
    from opentelemetry import context

    class _FailingTracer:
        def start_span(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("sampler exploded")

    fake_response = object()

    result = responses_wrappers._create_response_stream(
        tracer=_FailingTracer(),
        response=fake_response,
        start_time=0,
        request_kwargs={"model": "gpt-4.1-nano", "input": "hi"},
        instance=None,
        context=context.get_current(),
    )

    assert result is fake_response


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming_async")
@pytest.mark.asyncio
@pytest.mark.parametrize("stream_type", [None, CustomAsyncResponseStream])
async def test_async_responses_with_raw_response_streaming_does_not_crash(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
    stream_type: type[CustomAsyncResponseStream] | None,
) -> None:
    """Regression test for https://github.com/traceloop/openllmetry/issues/4476:
    client.responses.with_raw_response.create(stream=True, ...) (what
    agent-framework-openai>=1.6 uses to read response headers before streaming) must
    not crash and must emit a span after the returned stream is consumed."""
    raw = await async_openai_client.responses.with_raw_response.create(
        model="gpt-4.1-nano",
        input=_STORY_PROMPT,
        reasoning=None,
        stream=True,
    )
    # The raw-response contract (what agent-framework-openai relies on: reading a
    # response header before consuming the stream) must survive instrumentation.
    assert raw.headers["openai-version"] == "2020-10-01"

    stream = raw.parse(to=stream_type)
    assert raw.parse(to=stream_type) is stream
    if stream_type is not None:
        assert isinstance(stream, stream_type)
    events = [event async for event in stream]

    _assert_event_types(events, stream_type)
    _assert_streamed_span(span_exporter, events)


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming")
@pytest.mark.parametrize("stream_type", [None, CustomResponseStream])
def test_responses_with_raw_response_streaming_does_not_crash(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
    stream_type: type[CustomResponseStream] | None,
) -> None:
    """Sync counterpart of test_async_responses_with_raw_response_streaming_does_not_crash."""
    raw = openai_client.responses.with_raw_response.create(
        model="gpt-4.1-nano",
        input=_STORY_PROMPT,
        reasoning=None,
        stream=True,
    )
    # The raw-response contract (what agent-framework-openai relies on: reading a
    # response header before consuming the stream) must survive instrumentation.
    assert raw.headers["openai-version"] == "2020-10-01"

    stream = raw.parse(to=stream_type)
    assert raw.parse(to=stream_type) is stream
    if stream_type is not None:
        assert isinstance(stream, stream_type)
    events = list(stream)

    _assert_event_types(events, stream_type)
    _assert_streamed_span(span_exporter, events)


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming")
def test_responses_with_raw_response_parsed_outside_parent_span(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    openai_client: OpenAI,
) -> None:
    """The span joins the trace that was active when the request was made, even if
    raw.parse() runs after that parent span has ended."""
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("raw-response-parent") as parent_span:
        raw = openai_client.responses.with_raw_response.create(
            model="gpt-4.1-nano",
            input=_STORY_PROMPT,
            stream=True,
        )

    events = list(raw.parse())

    span = _assert_streamed_span(span_exporter, events)
    parent_context = parent_span.get_span_context()
    assert span.context.trace_id == parent_context.trace_id
    assert span.parent.span_id == parent_context.span_id


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming_async")
@pytest.mark.asyncio
async def test_async_responses_with_raw_response_parsed_outside_parent_span(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    async_openai_client: AsyncOpenAI,
) -> None:
    """Async counterpart of test_responses_with_raw_response_parsed_outside_parent_span."""
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("raw-response-parent") as parent_span:
        raw = await async_openai_client.responses.with_raw_response.create(
            model="gpt-4.1-nano",
            input=_STORY_PROMPT,
            stream=True,
        )

    events = [event async for event in raw.parse()]

    span = _assert_streamed_span(span_exporter, events)
    parent_context = parent_span.get_span_context()
    assert span.context.trace_id == parent_context.trace_id
    assert span.parent.span_id == parent_context.span_id


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming")
def test_raw_response_parse_with_two_types_emits_one_span(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
) -> None:
    """Regression test: the SDK caches raw.parse() per `to` type (LegacyAPIResponse.parse in
    openai 1.99.7 keys its cache on `to`), so a caller who also calls the plain raw.parse() on
    the same raw response -- e.g. a framework that peeks at the default type before choosing
    a custom one -- invokes our post_parser hook once per type for one HTTP response. Draining
    the custom stream and merely closing the unused default view used to export a second,
    mostly empty openai.response span."""
    raw = openai_client.responses.with_raw_response.create(
        model="gpt-4.1-nano",
        input=_STORY_PROMPT,
        stream=True,
    )

    custom_stream = raw.parse(to=CustomResponseStream)
    default_stream = raw.parse()
    assert default_stream is not custom_stream

    events = list(custom_stream)
    default_stream.close()

    _assert_streamed_span(span_exporter, events)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming_async")
@pytest.mark.asyncio
async def test_async_raw_response_parse_with_two_types_emits_one_span(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
) -> None:
    """Async counterpart of test_raw_response_parse_with_two_types_emits_one_span."""
    raw = await async_openai_client.responses.with_raw_response.create(
        model="gpt-4.1-nano",
        input=_STORY_PROMPT,
        stream=True,
    )

    custom_stream = raw.parse(to=CustomAsyncResponseStream)
    default_stream = raw.parse()
    assert default_stream is not custom_stream

    events = [event async for event in custom_stream]
    await default_stream.aclose()

    _assert_streamed_span(span_exporter, events)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming")
def test_raw_response_parse_default_first_still_traces_consumed_custom_stream(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
) -> None:
    """Regression test: with the default type parsed first and the custom type parsed
    and actually consumed second, the span must still carry the custom stream's data.
    Giving the span to only the first stream a caller's raw.parse() returns would leave
    the one the caller actually reads untraced, and end an empty span as soon as the
    unused first view is closed."""
    raw = openai_client.responses.with_raw_response.create(
        model="gpt-4.1-nano",
        input=_STORY_PROMPT,
        stream=True,
    )

    default_stream = raw.parse()
    custom_stream = raw.parse(to=CustomResponseStream)
    assert default_stream is not custom_stream

    events = list(custom_stream)
    default_stream.close()

    _assert_streamed_span(span_exporter, events)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming_async")
@pytest.mark.asyncio
async def test_async_raw_response_parse_default_first_still_traces_consumed_custom_stream(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
) -> None:
    """Async counterpart of test_raw_response_parse_default_first_still_traces_consumed_custom_stream."""
    raw = await async_openai_client.responses.with_raw_response.create(
        model="gpt-4.1-nano",
        input=_STORY_PROMPT,
        stream=True,
    )

    default_stream = raw.parse()
    custom_stream = raw.parse(to=CustomAsyncResponseStream)
    assert default_stream is not custom_stream

    events = [event async for event in custom_stream]
    await default_stream.aclose()

    _assert_streamed_span(span_exporter, events)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming")
def test_raw_response_span_reflects_request_sent_not_later_mutations(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
) -> None:
    """Regression test: the span must describe the request that was actually sent, not
    whatever the caller's kwargs look like by the time raw.parse() gets around to reading
    them. request_kwargs used to be stored by reference (only sentinel-filtered with a
    shallow dict copy) and read later from trace_stream, so a caller mutating nested request
    content -- e.g. appending to its own `input` list -- between create() and parse() changed
    what the span reported."""
    input_messages = [{"role": "user", "content": _STORY_PROMPT}]
    raw = openai_client.responses.with_raw_response.create(
        model="gpt-4.1-nano",
        input=input_messages,
        stream=True,
    )
    # The request has already been sent; mutate the caller's own list in place before
    # raw.parse() installs the traced stream and reads `input` for the span.
    input_messages[0]["content"] = "mutated after the request was sent"
    input_messages.append({"role": "user", "content": "a second message added too late"})

    events = list(raw.parse())

    span = _assert_streamed_span(span_exporter, events)
    input_json = span.attributes["gen_ai.input.messages"]
    assert "mutated after the request was sent" not in input_json
    assert "a second message added too late" not in input_json


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming")
def test_responses_streaming_with_reasoning_none(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
) -> None:
    """reasoning=None used to raise AttributeError from create(stream=True)."""
    stream = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=_STORY_PROMPT,
        reasoning=None,
        stream=True,
    )

    _assert_streamed_span(span_exporter, list(stream))


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming_async")
@pytest.mark.asyncio
async def test_responses_streaming_with_reasoning_none_async(
    instrument_legacy, span_exporter: InMemorySpanExporter, async_openai_client: AsyncOpenAI
) -> None:
    """Async counterpart of test_responses_streaming_with_reasoning_none."""
    stream = await async_openai_client.responses.create(
        model="gpt-4.1-nano",
        input=_STORY_PROMPT,
        reasoning=None,
        stream=True,
    )

    _assert_streamed_span(span_exporter, [event async for event in stream])


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses")
def test_responses_with_reasoning_none(
    clean_responses_registry,
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
) -> None:
    """reasoning=None used to drop the span of a non-streaming create()."""
    response = openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        reasoning=None,
    )

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "openai.response"
    assert get_output_messages(span)[0]["parts"][0]["content"] == response.output_text


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses")
@pytest.mark.asyncio
async def test_responses_with_reasoning_none_async(
    clean_responses_registry,
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
) -> None:
    """Async counterpart of test_responses_with_reasoning_none."""
    response = await async_openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        reasoning=None,
    )

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "openai.response"
    assert get_output_messages(span)[0]["parts"][0]["content"] == response.output_text


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming")
@pytest.mark.parametrize("raw_response", [False, True])
def test_stream_tracing_failure_returns_untraced_stream(
    instrument_legacy,
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
    raw_response: bool,
) -> None:
    """A tracing failure must not reach the caller, directly or through raw.parse()."""
    monkeypatch.setattr(responses_wrappers, "ResponseStream", _fail_stream_proxy)
    responses = (
        openai_client.responses.with_raw_response if raw_response else openai_client.responses
    )

    result = responses.create(model="gpt-4.1-nano", input=_STORY_PROMPT, stream=True)
    stream = result.parse() if raw_response else result

    _assert_tracing_failure_span(span_exporter, list(stream))


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming_async")
@pytest.mark.asyncio
@pytest.mark.parametrize("raw_response", [False, True])
async def test_stream_tracing_failure_returns_untraced_stream_async(
    instrument_legacy,
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
    raw_response: bool,
) -> None:
    """Async counterpart of test_stream_tracing_failure_returns_untraced_stream."""
    monkeypatch.setattr(responses_wrappers, "ResponseStream", _fail_stream_proxy)
    responses = (
        async_openai_client.responses.with_raw_response
        if raw_response
        else async_openai_client.responses
    )

    result = await responses.create(model="gpt-4.1-nano", input=_STORY_PROMPT, stream=True)
    stream = result.parse() if raw_response else result

    _assert_tracing_failure_span(span_exporter, [event async for event in stream])


@pytest.mark.vcr
@pytest.mark.default_cassette("test_responses_streaming")
def test_raw_response_without_sdk_options_is_left_untraced(
    instrument_legacy,
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
) -> None:
    """The instrumentation reads the SDK's private _options; without it create() still
    returns the raw response, untraced."""
    from openai._legacy_response import LegacyAPIResponse

    original_init = LegacyAPIResponse.__init__

    def init_without_options(self: LegacyAPIResponse, **kwargs: object) -> None:
        original_init(self, **kwargs)
        del self._options

    monkeypatch.setattr(LegacyAPIResponse, "__init__", init_without_options)

    raw = openai_client.responses.with_raw_response.create(
        model="gpt-4.1-nano",
        input=_STORY_PROMPT,
        stream=True,
    )

    assert raw.headers["openai-version"] == "2020-10-01"
    assert span_exporter.get_finished_spans() == ()


@pytest.fixture
def clean_responses_registry():
    """Reset the module-level `responses` dict and completed-ID cache around a test."""
    responses_wrappers.responses.clear()
    responses_wrappers._completed_response_ids.clear()
    yield
    responses_wrappers.responses.clear()
    responses_wrappers._completed_response_ids.clear()


def _make_fake_response(
    response_id="resp_test_1",
    status="completed",
    model="gpt-4.1-nano-2025-04-14",
    output_text="A fake reply.",
):
    """Minimal stand-in for an OpenAI `Response`.

    A SimpleNamespace rather than a MagicMock: MagicMock invents every attribute that is read
    (e.g. usage.input_tokens_details), which fails TracedData validation; the wrapper swallows
    that and returns early, so a test could pass without exercising anything. usage=None and
    output=[] likewise keep the real SDK types out of TracedData.
    """
    return types.SimpleNamespace(
        id=response_id,
        status=status,
        model=model,
        usage=None,
        output=[],
        output_text=output_text,
        service_tier=None,
        incomplete_details=None,
    )


def test_completed_sync_response_is_removed_from_responses_dict(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Regression test for https://github.com/traceloop/openllmetry/issues/4473:
    a completed response must be removed from the global `responses` dict once its span is emitted.
    """
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id="resp_sync_completed")

    def wrapped(*args, **kwargs):
        return fake_response

    responses_get_or_create_wrapper(tracer)(
        wrapped, None, (), {"model": "gpt-4.1-nano", "input": "hi"}
    )

    assert len(span_exporter.get_finished_spans()) == 1
    assert "resp_sync_completed" not in responses_wrappers.responses
    assert len(responses_wrappers.responses) == 0


@pytest.mark.asyncio
async def test_completed_async_response_is_removed_from_responses_dict(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Async counterpart of test_completed_sync_response_is_removed_from_responses_dict (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id="resp_async_completed")

    async def wrapped(*args, **kwargs):
        return fake_response

    await async_responses_get_or_create_wrapper(tracer)(
        wrapped, None, (), {"model": "gpt-4.1-nano", "input": "hi"}
    )

    assert len(span_exporter.get_finished_spans()) == 1
    assert "resp_async_completed" not in responses_wrappers.responses
    assert len(responses_wrappers.responses) == 0


class _FakeSyncChunkStream:
    """Minimal stand-in for the raw SDK stream that ResponseStream wraps."""

    def __init__(self, chunks):
        self._iter = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


def test_completed_streaming_response_is_removed_from_responses_dict(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """A completed stream must not leave an entry in `responses` (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id="resp_stream_completed")
    chunks = [
        types.SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        types.SimpleNamespace(type="response.completed", response=fake_response),
    ]
    span = tracer.start_span("openai.response")

    stream = ResponseStream(
        span=span,
        response=_FakeSyncChunkStream(chunks),
        start_time=0,
        request_kwargs={"model": "gpt-4.1-nano", "input": "hi"},
        tracer=tracer,
    )

    list(stream)

    assert len(span_exporter.get_finished_spans()) == 1
    assert "resp_stream_completed" not in responses_wrappers.responses
    assert len(responses_wrappers.responses) == 0


class _FakeAsyncChunkStream:
    """Async counterpart of _FakeSyncChunkStream."""

    def __init__(self, chunks):
        self._iter = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def test_streaming_output_text_preserved_when_terminal_event_is_filtered(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Regression test: a custom raw.parse(to=...) stream can filter out response.completed
    while still forwarding response.output_text.delta (see CustomResponseStream above,
    _FORWARDED_EVENT_TYPES). The span must still report the text we actually observed
    instead of silently leaving gen_ai.output.messages empty just because the terminal
    event never arrived."""
    tracer = tracer_provider.get_tracer(__name__)
    chunks = [
        types.SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        types.SimpleNamespace(type="response.output_text.delta", delta=" world"),
    ]
    stream = ResponseStream(
        span=tracer.start_span("openai.response"),
        response=_FakeSyncChunkStream(chunks),
        start_time=0,
        request_kwargs={"model": "gpt-4.1-nano", "input": "hi"},
        tracer=tracer,
    )

    list(stream)

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    assert get_output_messages(span)[0]["parts"][0]["content"] == "Hello world"


def test_closing_the_returned_iterator_ends_the_span(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Regression test: __iter__ is a generator function (see its docstring: not `return
    self`), so iter(stream) returns a plain generator object rather than the ResponseStream
    proxy itself. Closing that generator directly -- `it = iter(stream); next(it);
    it.close()` -- throws GeneratorExit at the suspended yield. GeneratorExit is not an
    Exception subclass, so it used to skip both the except Exception branch and the else
    exhaustion path below, leaving the span open even though calling stream.close() (or
    stream.__next__(), a with block, or dropping the stream) already ended it correctly."""
    tracer = tracer_provider.get_tracer(__name__)
    chunks = [
        types.SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        types.SimpleNamespace(type="response.output_text.delta", delta=" world"),
    ]
    stream = ResponseStream(
        span=tracer.start_span("openai.response"),
        response=_FakeSyncChunkStream(chunks),
        start_time=0,
        request_kwargs={"model": "gpt-4.1-nano", "input": "hi"},
        tracer=tracer,
    )

    iterator = iter(stream)
    next(iterator)
    iterator.close()

    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_closing_the_returned_async_iterator_ends_the_span(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Async counterpart of test_closing_the_returned_iterator_ends_the_span: aclose() on
    the async generator that __aiter__ returns must still end the span."""
    tracer = tracer_provider.get_tracer(__name__)
    chunks = [
        types.SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        types.SimpleNamespace(type="response.output_text.delta", delta=" world"),
    ]
    stream = ResponseStream(
        span=tracer.start_span("openai.response"),
        response=_FakeAsyncChunkStream(chunks),
        start_time=0,
        request_kwargs={"model": "gpt-4.1-nano", "input": "hi"},
        tracer=tracer,
    )

    iterator = stream.__aiter__()
    await iterator.__anext__()
    await iterator.aclose()

    assert len(span_exporter.get_finished_spans()) == 1


class _FakeSyncChunkStreamThatRaises:
    """Raises instead of completing normally, simulating a custom raw.parse(to=...)
    stream's own iteration failing mid-stream."""

    def __init__(self, chunks, exception: Exception):
        self._iter = iter(chunks)
        self._exception = exception

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise self._exception from None


class _FakeAsyncChunkStreamThatRaises:
    """Async counterpart of _FakeSyncChunkStreamThatRaises."""

    def __init__(self, chunks, exception: Exception):
        self._iter = iter(chunks)
        self._exception = exception

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise self._exception from None


def test_shared_lifecycle_reports_error_from_a_sibling_that_closes_later(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Regression test: if the custom stream raises while the default stream (sharing
    the same span, see _SharedResponseStreamLifecycle) is still open, the erroring
    sibling can't end the span by itself; nothing latched that error, so the default
    stream closing cleanly afterward ended the shared span with OK instead of
    reporting it. The span must end with ERROR once every sibling is done, no matter
    which one happens to trigger that."""
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("openai.response")
    lifecycle = responses_wrappers._SharedResponseStreamLifecycle()
    request_kwargs = {"model": "gpt-4.1-nano", "input": "hi"}

    default_stream = ResponseStream(
        span=span,
        response=_FakeSyncChunkStream([]),
        start_time=0,
        request_kwargs=request_kwargs,
        tracer=tracer,
        shared_lifecycle=lifecycle,
    )
    boom = RuntimeError("custom stream blew up")
    custom_stream = ResponseStream(
        span=span,
        response=_FakeSyncChunkStreamThatRaises(
            [types.SimpleNamespace(type="response.output_text.delta", delta="partial")], boom
        ),
        start_time=0,
        request_kwargs=request_kwargs,
        tracer=tracer,
        traced_data=default_stream._traced_data,
        shared_lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError, match="custom stream blew up"):
        list(custom_stream)

    # The default stream is still open, so the erroring custom stream can't end the
    # shared span by itself yet.
    assert span_exporter.get_finished_spans() == ()

    default_stream.close()

    (finished_span,) = span_exporter.get_finished_spans()
    assert finished_span.status.status_code == StatusCode.ERROR
    assert finished_span.attributes["error.type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_async_shared_lifecycle_reports_error_from_a_sibling_that_closes_later(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Async counterpart of test_shared_lifecycle_reports_error_from_a_sibling_that_closes_later."""
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("openai.response")
    lifecycle = responses_wrappers._SharedResponseStreamLifecycle()
    request_kwargs = {"model": "gpt-4.1-nano", "input": "hi"}

    default_stream = ResponseStream(
        span=span,
        response=_FakeAsyncChunkStream([]),
        start_time=0,
        request_kwargs=request_kwargs,
        tracer=tracer,
        shared_lifecycle=lifecycle,
    )
    boom = RuntimeError("custom stream blew up")
    custom_stream = ResponseStream(
        span=span,
        response=_FakeAsyncChunkStreamThatRaises(
            [types.SimpleNamespace(type="response.output_text.delta", delta="partial")], boom
        ),
        start_time=0,
        request_kwargs=request_kwargs,
        tracer=tracer,
        traced_data=default_stream._traced_data,
        shared_lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError, match="custom stream blew up"):
        [event async for event in custom_stream]

    assert span_exporter.get_finished_spans() == ()

    await default_stream.aclose()

    (finished_span,) = span_exporter.get_finished_spans()
    assert finished_span.status.status_code == StatusCode.ERROR
    assert finished_span.attributes["error.type"] == "RuntimeError"


def test_duplicate_retrieve_after_completion_does_not_emit_second_span(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """retrieve() on an already-completed, cleaned-up response must not emit a second span
    rebuilt without the original input, tools and trace context (#4473).
    """
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id="resp_dup_retrieve")

    def wrapped(*args, **kwargs):
        return fake_response

    responses_get_or_create_wrapper(tracer)(
        wrapped, None, (), {"model": "gpt-4.1-nano", "input": "hi"}
    )
    responses_get_or_create_wrapper(tracer)(
        wrapped, None, (), {"response_id": "resp_dup_retrieve"}
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, "a duplicate retrieve on a completed response must not emit a second span"
    assert len(responses_wrappers.responses) == 0


@pytest.mark.asyncio
async def test_async_duplicate_retrieve_after_completion_does_not_emit_second_span(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Async counterpart of test_duplicate_retrieve_after_completion_does_not_emit_second_span (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id="resp_async_dup_retrieve")

    async def wrapped(*args, **kwargs):
        return fake_response

    await async_responses_get_or_create_wrapper(tracer)(
        wrapped, None, (), {"model": "gpt-4.1-nano", "input": "hi"}
    )
    await async_responses_get_or_create_wrapper(tracer)(
        wrapped, None, (), {"response_id": "resp_async_dup_retrieve"}
    )

    assert len(span_exporter.get_finished_spans()) == 1
    assert len(responses_wrappers.responses) == 0


def test_completed_response_cache_is_bounded(clean_responses_registry, monkeypatch):
    """The completed-ID cache evicts its oldest IDs instead of growing without bound (#4473)."""
    monkeypatch.setattr(responses_wrappers, "_MAX_TRACKED_COMPLETED_RESPONSES", 3)
    responses_wrappers._completed_response_ids.clear()

    for i in range(5):
        responses_wrappers._record_traced_data(
            f"resp_{i}", responses_wrappers.TracedData(start_time=0, response_id=f"resp_{i}", input="hi"), True
        )

    assert len(responses_wrappers._completed_response_ids) == 3
    assert not responses_wrappers._was_response_already_completed("resp_0")
    assert not responses_wrappers._was_response_already_completed("resp_1")
    assert responses_wrappers._was_response_already_completed("resp_4")


def test_retrieve_after_completed_stream_does_not_emit_second_span(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """retrieve() after a completed stream must not emit a duplicate span (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id="resp_stream_then_retrieve")
    chunks = [types.SimpleNamespace(type="response.completed", response=fake_response)]
    stream = ResponseStream(
        span=tracer.start_span("openai.response"),
        response=_FakeSyncChunkStream(chunks),
        start_time=0,
        request_kwargs={"model": "gpt-4.1-nano", "input": "hi"},
        tracer=tracer,
    )
    list(stream)

    responses_get_or_create_wrapper(tracer)(
        lambda *a, **kw: fake_response, None, (), {"response_id": "resp_stream_then_retrieve"}
    )

    assert len(span_exporter.get_finished_spans()) == 1
    assert len(responses_wrappers.responses) == 0


def test_polled_response_emits_one_span_with_original_input_then_is_removed(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Polling in_progress -> completed emits one span with the create() input, then drops the entry (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    in_progress = _make_fake_response(response_id="resp_polled", status="in_progress")
    completed = _make_fake_response(response_id="resp_polled", status="completed")

    responses_get_or_create_wrapper(tracer)(
        lambda *a, **kw: in_progress, None, (), {"model": "gpt-4.1-nano", "input": "original question"}
    )
    between_create_and_retrieve = time.time_ns()
    responses_get_or_create_wrapper(tracer)(
        lambda *a, **kw: completed, None, (), {"response_id": "resp_polled"}
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "original question" in spans[0].attributes["gen_ai.input.messages"]
    assert spans[0].start_time < between_create_and_retrieve, "span must start at create(), not retrieve()"
    assert len(responses_wrappers.responses) == 0


def test_many_completed_turns_leave_responses_dict_empty(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Repeated completed turns, as in a chat workload, must not accumulate entries in `responses` (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    turns = 100

    for i in range(turns):
        fake_response = _make_fake_response(response_id=f"resp_turn_{i}")
        responses_get_or_create_wrapper(tracer)(
            lambda *a, _r=fake_response, **kw: _r, None, (), {"model": "gpt-4.1-nano", "input": f"turn {i}"}
        )
        assert len(responses_wrappers.responses) == 0

    assert len(span_exporter.get_finished_spans()) == turns


def test_retrieve_after_interrupted_stream_keeps_original_input(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """A background stream left before completing must not block the later completed
    retrieve() span, and keeps its entry so that span still has the original input (#4473).
    """
    tracer = tracer_provider.get_tracer(__name__)
    in_progress = _make_fake_response(response_id="resp_bg_interrupted", status="in_progress")
    completed = _make_fake_response(response_id="resp_bg_interrupted", status="completed")
    stream = ResponseStream(
        span=tracer.start_span("openai.response"),
        response=_FakeSyncChunkStream(
            [types.SimpleNamespace(type="response.in_progress", response=in_progress)]
        ),
        start_time=0,
        request_kwargs={"model": "gpt-4.1-nano", "input": "original question", "background": True},
        tracer=tracer,
    )
    list(stream)

    responses_get_or_create_wrapper(tracer)(
        lambda *a, **kw: completed, None, (), {"response_id": "resp_bg_interrupted"}
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2, "the completed retrieve() must emit its own span after the stream span"
    assert "original question" in spans[-1].attributes["gen_ai.input.messages"]
    assert len(responses_wrappers.responses) == 0


def test_concurrent_completed_retrieves_emit_one_span(
    clean_responses_registry,
    monkeypatch, span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Two threads completing the same response at once must emit only one span (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id="resp_concurrent")
    both_emitting = threading.Barrier(2)
    original_set_data_attributes = responses_wrappers.set_data_attributes

    def set_data_attributes_waiting_for_other_thread(traced_data, span):
        try:
            both_emitting.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        original_set_data_attributes(traced_data, span)

    monkeypatch.setattr(responses_wrappers, "set_data_attributes", set_data_attributes_waiting_for_other_thread)

    def retrieve():
        responses_get_or_create_wrapper(tracer)(
            lambda *a, **kw: fake_response, None, (), {"model": "gpt-4.1-nano", "input": "hi"}
        )

    threads = [threading.Thread(target=retrieve) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(span_exporter.get_finished_spans()) == 1
    assert len(responses_wrappers.responses) == 0


def test_delayed_in_progress_poll_does_not_restore_entry_after_completion(
    clean_responses_registry,
    monkeypatch, span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """A slow in_progress poll that read the entry before another poll completed and
    removed it must not write its stale data back afterwards (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    in_progress = _make_fake_response(response_id="resp_delayed_poll", status="in_progress")
    completed = _make_fake_response(response_id="resp_delayed_poll", status="completed")
    responses_get_or_create_wrapper(tracer)(
        lambda *a, **kw: in_progress, None, (), {"model": "gpt-4.1-nano", "input": "hi"}
    )

    delayed_read_entry = threading.Event()
    completion_done = threading.Event()
    original_get_tools_from_kwargs = responses_wrappers.get_tools_from_kwargs

    def get_tools_pausing_delayed_poll(kwargs):
        # Runs after the wrapper has read `responses` and before it writes back.
        if threading.current_thread().name == "delayed-poll":
            delayed_read_entry.set()
            completion_done.wait(timeout=2)
        return original_get_tools_from_kwargs(kwargs)

    monkeypatch.setattr(responses_wrappers, "get_tools_from_kwargs", get_tools_pausing_delayed_poll)

    delayed_poll = threading.Thread(
        name="delayed-poll",
        target=lambda: responses_get_or_create_wrapper(tracer)(
            lambda *a, **kw: in_progress, None, (), {"response_id": "resp_delayed_poll"}
        ),
    )
    delayed_poll.start()
    assert delayed_read_entry.wait(timeout=2)
    responses_get_or_create_wrapper(tracer)(
        lambda *a, **kw: completed, None, (), {"response_id": "resp_delayed_poll"}
    )
    completion_done.set()
    delayed_poll.join(timeout=2)

    assert len(span_exporter.get_finished_spans()) == 1
    assert "resp_delayed_poll" not in responses_wrappers.responses, "stale in_progress data was restored"


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
def test_terminal_sync_response_is_removed_and_emits_span(
    status, clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """incomplete, failed and cancelled are terminal too: the entry is freed and a span emitted (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id=f"resp_sync_{status}", status=status)

    responses_get_or_create_wrapper(tracer)(
        lambda *a, **kw: fake_response, None, (), {"model": "gpt-4.1-nano", "input": "hi"}
    )

    assert len(span_exporter.get_finished_spans()) == 1
    assert len(responses_wrappers.responses) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
async def test_terminal_async_response_is_removed_and_emits_span(
    status, clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Async counterpart of test_terminal_sync_response_is_removed_and_emits_span (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id=f"resp_async_{status}", status=status)

    async def wrapped(*args, **kwargs):
        return fake_response

    await async_responses_get_or_create_wrapper(tracer)(
        wrapped, None, (), {"model": "gpt-4.1-nano", "input": "hi"}
    )

    assert len(span_exporter.get_finished_spans()) == 1
    assert len(responses_wrappers.responses) == 0


def test_incomplete_streaming_response_is_removed_from_responses_dict(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """A stream ending in response.incomplete must not leave an entry in `responses` (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    fake_response = _make_fake_response(response_id="resp_stream_incomplete", status="incomplete")
    chunks = [
        types.SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        types.SimpleNamespace(type="response.incomplete", response=fake_response),
    ]

    stream = ResponseStream(
        span=tracer.start_span("openai.response"),
        response=_FakeSyncChunkStream(chunks),
        start_time=0,
        request_kwargs={"model": "gpt-4.1-nano", "input": "hi"},
        tracer=tracer,
    )
    list(stream)

    assert len(span_exporter.get_finished_spans()) == 1
    assert len(responses_wrappers.responses) == 0


def test_retrieve_after_cancel_does_not_restore_entry_or_emit_second_span(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """cancel() emits the span and frees the entry; a later retrieve() returning cancelled
    must not write the entry back or emit a second span (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    in_progress = _make_fake_response(response_id="resp_cancel", status="in_progress")
    cancelled = _make_fake_response(response_id="resp_cancel", status="cancelled")

    responses_get_or_create_wrapper(tracer)(
        lambda *a, **kw: in_progress, None, (), {"model": "gpt-4.1-nano", "input": "hi", "background": True}
    )
    responses_cancel_wrapper(tracer)(
        lambda *a, **kw: cancelled, None, (), {"response_id": "resp_cancel"}
    )
    responses_get_or_create_wrapper(tracer)(
        lambda *a, **kw: cancelled, None, (), {"response_id": "resp_cancel"}
    )

    assert len(span_exporter.get_finished_spans()) == 1
    assert len(responses_wrappers.responses) == 0


def test_non_background_stream_left_early_is_removed_from_responses_dict(
    clean_responses_registry,
    span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
):
    """Leaving a non-background stream early must not keep its in_progress entry: only
    background=True responses can be retrieved later to merge the original request (#4473)."""
    tracer = tracer_provider.get_tracer(__name__)
    in_progress = _make_fake_response(response_id="resp_stream_left_early", status="in_progress")
    chunks = [
        types.SimpleNamespace(type="response.created", response=in_progress),
        types.SimpleNamespace(type="response.output_text.delta", delta="Hello"),
    ]

    with ResponseStream(
        span=tracer.start_span("openai.response"),
        response=_FakeSyncChunkStream(chunks),
        start_time=0,
        request_kwargs={"model": "gpt-4.1-nano", "input": "hi"},
        tracer=tracer,
    ) as stream:
        for _ in stream:
            break

    assert len(span_exporter.get_finished_spans()) == 1
    assert len(responses_wrappers.responses) == 0
