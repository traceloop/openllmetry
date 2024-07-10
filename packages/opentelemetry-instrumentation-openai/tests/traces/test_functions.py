import json
import pytest
from opentelemetry.semconv.ai import SpanAttributes


@pytest.fixture
def openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
    ]


@pytest.mark.vcr
def test_open_ai_function_calls(exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "What's the weather like in Boston?"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description"]
        == "Get the current weather in a given location"
    )

    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.type"]
        == "function"
    )
    assert (
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.name"
        ]
        == "get_current_weather"
    )
    assert json.loads(
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.arguments"
        ]
    ) == {"location": "Boston"}

    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )


@pytest.mark.vcr
def test_open_ai_function_calls_tools(exporter, openai_client, openai_tools):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        tools=openai_tools,
        tool_choice="auto",
    )

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "What's the weather like in Boston?"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description"]
        == "Get the current weather"
    )

    assert isinstance(
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id"],
        str,
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.type"]
        == "function"
    )
    assert (
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.name"
        ]
        == "get_current_weather"
    )
    assert json.loads(
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.arguments"
        ]
    ) == {"location": "Boston"}

    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming(
    exporter, async_openai_client, openai_tools
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == "get_current_weather"
    )

    assert isinstance(
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id"],
        str,
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.type"]
        == "function"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.arguments"
        )
        == '{"location":"San Francisco, CA"}'
    )


@pytest.mark.vcr
def test_open_ai_function_calls_tools_parallel(exporter, openai_client, openai_tools):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
    )

    for _ in response:
        pass

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == "get_current_weather"
    )

    assert isinstance(
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id"],
        str,
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.type"]
        == "function"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.arguments"
        )
        == '{"location": "San Francisco"}'
    )

    assert isinstance(
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.id"],
        str,
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.type"]
        == "function"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.function.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.function.arguments"
        )
        == '{"location": "Boston"}'
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming_parallel(
    exporter, async_openai_client, openai_tools
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == "get_current_weather"
    )

    assert isinstance(
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id"],
        str,
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.type"]
        == "function"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.arguments"
        )
        == '{"location": "San Francisco"}'
    )

    assert isinstance(
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.id"],
        str,
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.type"]
        == "function"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.function.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.function.arguments"
        )
        == '{"location": "Boston"}'
    )
