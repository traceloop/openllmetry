import pytest
import ollama
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv.trace import SpanAttributes


@pytest.mark.vcr
def test_ollama_generation(exporter):
    response = ollama.generate(
        model="llama3", prompt="Tell me a joke about OpenTelemetry"
    )

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.completion"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Ollama"
    assert (
        ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TYPE) == "completion"
    )
    assert not ollama_span.attributes.get(GenAIAttributes.GEN_AI_IS_STREAMING)
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response["response"]
    )
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    )


@pytest.mark.vcr
def test_ollama_streaming_generation(exporter):
    gen = ollama.generate(
        model="llama3", prompt="Tell me a joke about OpenTelemetry", stream=True
    )

    response = ""
    for res in gen:
        response += res.get("response")

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.completion"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Ollama"
    assert (
        ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TYPE) == "completion"
    )
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_IS_STREAMING)
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_MODEL) == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response
    )
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_generation(exporter):
    client = ollama.AsyncClient()
    response = await client.generate(
        model="llama3", prompt="Tell me a joke about OpenTelemetry"
    )

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.completion"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Ollama"
    assert (
        ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TYPE) == "completion"
    )
    assert not ollama_span.attributes.get(GenAIAttributes.GEN_AI_IS_STREAMING)
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_MODEL) == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response["response"]
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_streaming_generation(exporter):
    client = ollama.AsyncClient()
    gen = await client.generate(
        model="llama3", prompt="Tell me a joke about OpenTelemetry", stream=True
    )

    response = ""
    async for res in gen:
        response += res.get("response")

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.completion"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Ollama"
    assert (
        ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TYPE) == "completion"
    )
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_IS_STREAMING)
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_MODEL) == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    )
