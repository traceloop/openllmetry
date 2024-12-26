import os

import httpx
import pytest
from unittest.mock import patch
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv.trace import SpanAttributes
from .utils import spy_decorator, assert_request_contains_tracecontext


@pytest.mark.vcr
def test_completion(exporter, openai_client):
    openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == "OpenAI"
    )
    assert open_ai_span.attributes.get(GenAIAttributes.GEN_AI_IS_STREAMING) is False


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion(exporter, async_openai_client):
    await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")


@pytest.mark.vcr
def test_completion_langchain_style(exporter, openai_client):
    openai_client.completions.create(
        model="davinci-002",
        prompt=["Tell me a joke about opentelemetry"],
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")


@pytest.mark.vcr
def test_completion_streaming(exporter, openai_client):
    # set os env for token usage record in stream mode
    original_value = os.environ.get("TRACELOOP_STREAM_TOKEN_USAGE")
    os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = "true"

    try:
        response = openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
            stream=True,
        )

        for _ in response:
            pass

        spans = exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            "openai.completion",
        ]
        open_ai_span = spans[0]
        assert (
            open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
            == "Tell me a joke about opentelemetry"
        )
        assert open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
        )
        assert (
            open_ai_span.attributes.get(GenAIAttributes.GEN_AI_API_BASE)
            == "https://api.openai.com/v1/"
        )

        # check token usage attributes for stream
        completion_tokens = open_ai_span.attributes.get(
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
        )
        prompt_tokens = open_ai_span.attributes.get(
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
        )
        total_tokens = open_ai_span.attributes.get(
            GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS
        )
        assert completion_tokens and prompt_tokens and total_tokens
        assert completion_tokens + prompt_tokens == total_tokens
    finally:
        # unset env
        if original_value is None:
            del os.environ["TRACELOOP_STREAM_TOKEN_USAGE"]
        else:
            os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = original_value


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_streaming(exporter, async_openai_client):
    response = await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
        stream=True,
    )

    async for _ in response:
        pass

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(GenAIAttributes.GEN_AI_API_BASE)
        == "https://api.openai.com/v1/"
    )


@pytest.mark.vcr
def test_completion_context_propagation(exporter, vllm_openai_client):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.completions.create(
            # model="davinci-002",
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )
    send_spy.mock.assert_called_once()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_context_propagation(exporter, async_vllm_openai_client):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )
    send_spy.mock.assert_called_once()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)
