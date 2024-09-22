import os
import pytest
from mistralai.client import MistralClient
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_mistralai_chat(exporter):
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].message.content
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )


@pytest.mark.vcr
def test_mistralai_streaming_chat(exporter):
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    gen = client.chat_stream(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    for res in gen:
        response += res.choices[0].delta.content

    spans = exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_chat(exporter):
    client = MistralAsyncClient(api_key=os.environ["MISTRAL_API_KEY"])
    response = await client.chat(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].message.content
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_streaming_chat(exporter):
    client = MistralAsyncClient(api_key=os.environ["MISTRAL_API_KEY"])
    gen = await client.chat_stream(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    async for res in gen:
        response += res.choices[0].delta.content

    spans = exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert mistral_span.attributes.get("gen_ai.request.model") == "mistral-tiny"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )
