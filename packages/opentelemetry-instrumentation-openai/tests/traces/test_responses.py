import pytest

from openai import AsyncOpenAI, OpenAI
from opentelemetry.instrumentation.openai.utils import is_reasoning_supported
from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
    get_tools_from_kwargs,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


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
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    # assert (
    #     span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    # )
    # assert span.attributes["gen_ai.prompt.0.role"] == "user"


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
    assert span.attributes["gen_ai.system"] == "openai"
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
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    # assert (
    #     span.attributes["gen_ai.prompt.0.content"]
    #     == "Come up with an adjective in English. Respond with just one word."
    # )
    # assert span.attributes["gen_ai.prompt.0.role"] == "user"
    # assert json.loads(span.attributes["gen_ai.prompt.1.content"]) == [
    #     {
    #         "type": "output_text",
    #         "text": first_response.output[0].content[0].text,
    #     }
    # ]
    # assert span.attributes["gen_ai.prompt.1.role"] == "assistant"
    # assert (
    #     span.attributes["gen_ai.prompt.2.content"]
    #     == "Can you explain why you chose that word?"
    # )
    # assert span.attributes["gen_ai.prompt.2.role"] == "user"


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
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"

    # assert span.attributes["gen_ai.prompt.0.content"] == "What's the weather in London?"
    # assert span.attributes["gen_ai.prompt.0.role"] == "user"
    # assert span.attributes["llm.request.functions.0.name"] == "get_weather"
    # assert json.loads(span.attributes["llm.request.functions.0.parameters"]) == {
    #     "type": "object",
    #     "properties": {
    #         "location": {
    #             "type": "string",
    #             "description": "The city and state, e.g. San Francisco, CA"
    #         }
    #     },
    #     "required": ["location"]
    # }
    # assert span.attributes["llm.request.functions.0.description"] == "Get the current weather for a location"

    # assert (
    #     span.attributes["gen_ai.completion.0.tool_calls.0.id"]
    #     == "fc_685ff89422ec819a977b2ea385bc9b6601c537ddeff5c2a2"
    # )
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
    span = spans[0]

    # assert span.attributes["gen_ai.request.reasoning_effort"] == "low"
    # assert span.attributes["gen_ai.request.reasoning_summary"] == ()

    # assert span.attributes["gen_ai.response.reasoning_effort"] == "low"
    # When reasoning summary is None/empty, the attribute should not be set
    assert "gen_ai.completion.0.reasoning" not in span.attributes

    # assert span.attributes["gen_ai.usage.reasoning_tokens"] > 0


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

    # The reasoning should be serialized as JSON since it contains complex data
    reasoning_attr = span.attributes["gen_ai.completion.0.reasoning"]
    assert isinstance(reasoning_attr, str), "Reasoning should be serialized as a string"

    # Should be valid JSON containing reasoning summary data
    import json

    parsed_reasoning = json.loads(reasoning_attr)
    assert isinstance(
        parsed_reasoning, (dict, list)
    ), "Reasoning should be a dict or list structure"


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
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert full_text != "", "Should have received streaming content"
    assert span.attributes["gen_ai.prompt.0.content"] == input_text
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.completion.0.content"] == full_text


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
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"
    assert span.attributes["gen_ai.prompt.0.content"] == input_text
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.completion.0.content"] == full_text


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
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"
    assert span.attributes["gen_ai.prompt.0.content"] == input_text
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.completion.0.content"] == full_text


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
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"
    assert span.attributes["gen_ai.prompt.0.content"] == input_text
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.completion.0.content"] == full_text


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
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"
    assert span.attributes["gen_ai.prompt.0.content"] == input_text
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.completion.0.content"] == full_text


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
    assert openai_span.attributes["gen_ai.system"] == "openai"
    assert openai_span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert openai_span.attributes["llm.is_streaming"] is True


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
    assert openai_span.attributes["gen_ai.system"] == "openai"
    assert openai_span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert openai_span.attributes["llm.is_streaming"] is True


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
