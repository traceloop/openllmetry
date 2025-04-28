import pytest
from ollama import AsyncClient, chat
from opentelemetry.semconv_ai import SpanAttributes
from unittest.mock import MagicMock
from opentelemetry.instrumentation.ollama import _set_response_attributes
from opentelemetry.semconv_ai import LLMRequestTypeValues


@pytest.mark.vcr
def test_ollama_chat(exporter):
    response = chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
    )

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response["message"]["content"]
    )
    assert ollama_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )


@pytest.mark.vcr
def test_ollama_chat_tool_calls(exporter):
    chat(
        model="llama3.1",
        messages=[
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'function': {
                        'name': 'get_current_weather',
                        'arguments': '{"location": "San Francisco"}'
                    }
                }]
            },
            {
                'role': 'tool',
                'content': 'The weather in San Francisco is 70 degrees and sunny.'
            }
        ],
    )

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]

    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}") == "llama3.1"
    assert f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.content" not in ollama_span.attributes
    assert (
        ollama_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        ollama_span.attributes[
            f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.arguments"
        ]
        == '{"location": "San Francisco"}'
    )

    assert (
        ollama_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )


@pytest.mark.vcr
def test_ollama_streaming_chat(exporter):
    gen = chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        stream=True,
    )

    response = ""
    for res in gen:
        response += res["message"]["content"]

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )
    assert ollama_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_chat(exporter):
    client = AsyncClient()
    response = await client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
    )

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response["message"]["content"]
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert ollama_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_streaming_chat(exporter):
    client = AsyncClient()
    gen = await client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        stream=True,
    )

    response = ""
    async for res in gen:
        response += res["message"]["content"]

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )
    assert ollama_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )


@pytest.mark.vcr
def test_token_histogram_recording():
    span = MagicMock()
    token_histogram = MagicMock()
    llm_request_type = LLMRequestTypeValues.COMPLETION
    response = {
        "model": "llama3",
        "prompt_eval_count": 7,
        "eval_count": 10,
    }
    _set_response_attributes(span, token_histogram, llm_request_type, response)
    token_histogram.record.assert_any_call(
        7,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Ollama",
            SpanAttributes.LLM_TOKEN_TYPE: "input",
            SpanAttributes.LLM_RESPONSE_MODEL: "llama3",
        },
    )
    token_histogram.record.assert_any_call(
        10,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Ollama",
            SpanAttributes.LLM_TOKEN_TYPE: "output",
            SpanAttributes.LLM_RESPONSE_MODEL: "llama3",
        },
    )
