import asyncio
import httpx
import pytest
from unittest.mock import patch
from opentelemetry.semconv_ai import SpanAttributes
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
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get(
            SpanAttributes.LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
        )
        == "fp_2b778c6b35"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-908MD9ivBBLb6EaIjlqwFokntayQK"


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
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-9gKNZbUWSC4s2Uh2QfVV7PYiqWIuH"


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
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-9lvGJKrBUPeJjHi3KKSEbGfcfomOP"


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
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-908MECg5dMyTTbJEltubwQXeeWlBA"


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
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-9AGW3t9akkLW9f5f93B7mOhiqhNMC"


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
    open_ai_span = spans[0]
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-ANnyEsyt6uxfIIA7lcPLH95lKcEeK"


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
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chat-43f4347c3299481e9704ab77439fbdb8"
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
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chat-4db07f02ecae49cbafe1d359db1650df"
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)


@pytest.mark.vcr
def test_chat_context_propagation_v0(exporter, monkeypatch, openai_client):
    # Mock is_openai_v1 to return False for testing v0 behavior
    monkeypatch.setattr("opentelemetry.instrumentation.openai.utils.is_openai_v1", lambda: False)
    
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
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
    assert "headers" in request.headers  # Verify headers are used in v0


@pytest.mark.vcr
def test_chat_context_propagation_v1(exporter, monkeypatch, openai_client):
    # Mock is_openai_v1 to return True for testing v1 behavior
    monkeypatch.setattr("opentelemetry.instrumentation.openai.utils.is_openai_v1", lambda: True)
    
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
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
    assert "extra_headers" in request.headers  # Verify extra_headers are used in v1
