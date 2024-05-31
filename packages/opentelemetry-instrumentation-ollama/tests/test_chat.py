import pytest
import ollama
from opentelemetry.semconv.ai import SpanAttributes


@pytest.mark.vcr
def test_ollama_chat(exporter):
    response = ollama.chat(
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
def test_ollama_streaming_chat(exporter):
    gen = ollama.chat(
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
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response
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
    client = ollama.AsyncClient()
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
    client = ollama.AsyncClient()
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
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response
    assert ollama_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )
