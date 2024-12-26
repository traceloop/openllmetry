import asyncio
import httpx
import pytest
from unittest.mock import patch
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from .utils import spy_decorator, assert_request_contains_tracecontext


@pytest.mark.vcr
def test_chat(exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(GenAIAttributes.GEN_AI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get(
            GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
        )
        == "fp_2b778c6b35"
    )
    assert open_ai_span.attributes.get(GenAIAttributes.GEN_AI_IS_STREAMING) is False


@pytest.mark.vcr
def test_chat_tool_calls(exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.arguments"
        ]
        == '{"location": "San Francisco"}'
    )

    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "1"
    )


@pytest.mark.vcr
def test_chat_pydantic_based_tool_calls(exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    ChatCompletionMessageToolCall(
                        id="1",
                        type="function",
                        function=Function(
                            name="get_current_weather",
                            arguments='{"location": "San Francisco"}',
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.arguments"
        ]
        == '{"location": "San Francisco"}'
    )

    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "1"
    )


@pytest.mark.vcr
def test_chat_streaming(exporter, openai_client):

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    for _ in response:
        chunk_count += 1

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(GenAIAttributes.GEN_AI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(GenAIAttributes.GEN_AI_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming(exporter, async_openai_client):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    async for _ in response:
        chunk_count += 1

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(GenAIAttributes.GEN_AI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(GenAIAttributes.GEN_AI_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens


@pytest.mark.vcr
@pytest.mark.asyncio
def test_with_asyncio_run(exporter, async_openai_client):
    asyncio.run(
        async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        )
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]


@pytest.mark.vcr
def test_chat_context_propagation(exporter, vllm_openai_client):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        )
    send_spy.mock.assert_called_once()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_context_propagation(exporter, async_vllm_openai_client):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        )
    send_spy.mock.assert_called_once()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)
