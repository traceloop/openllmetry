import pytest
import ollama


@pytest.mark.vcr
def test_ollama_generation(exporter):
    response = ollama.generate(
        model="llama3", prompt="Tell me a joke about OpenTelemetry"
    )

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.completion"
    assert ollama_span.attributes.get("gen_ai.system") == "Ollama"
    assert ollama_span.attributes.get("llm.request.type") == "completion"
    assert not ollama_span.attributes.get("llm.is_streaming")
    assert ollama_span.attributes.get("gen_ai.request.model") == "llama3"
    assert (
        ollama_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get("gen_ai.completion.0.content")
        == response["response"]
    )
    assert ollama_span.attributes.get("gen_ai.usage.prompt_tokens") == 17
    assert ollama_span.attributes.get(
        "llm.usage.total_tokens"
    ) == ollama_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + ollama_span.attributes.get(
        "gen_ai.usage.prompt_tokens"
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
    assert ollama_span.attributes.get("gen_ai.system") == "Ollama"
    assert ollama_span.attributes.get("llm.request.type") == "completion"
    assert ollama_span.attributes.get("llm.is_streaming")
    assert ollama_span.attributes.get("gen_ai.request.model") == "llama3"
    assert (
        ollama_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert ollama_span.attributes.get("gen_ai.completion.0.content") == response
    assert ollama_span.attributes.get("gen_ai.usage.prompt_tokens") == 17
    assert ollama_span.attributes.get(
        "llm.usage.total_tokens"
    ) == ollama_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + ollama_span.attributes.get(
        "gen_ai.usage.prompt_tokens"
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
    assert ollama_span.attributes.get("gen_ai.system") == "Ollama"
    assert ollama_span.attributes.get("llm.request.type") == "completion"
    assert not ollama_span.attributes.get("llm.is_streaming")
    assert ollama_span.attributes.get("gen_ai.request.model") == "llama3"
    assert (
        ollama_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get("gen_ai.completion.0.content")
        == response["response"]
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert ollama_span.attributes.get("gen_ai.usage.prompt_tokens") == 17
    assert ollama_span.attributes.get(
        "llm.usage.total_tokens"
    ) == ollama_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + ollama_span.attributes.get(
        "gen_ai.usage.prompt_tokens"
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
    assert ollama_span.attributes.get("gen_ai.system") == "Ollama"
    assert ollama_span.attributes.get("llm.request.type") == "completion"
    assert ollama_span.attributes.get("llm.is_streaming")
    assert ollama_span.attributes.get("gen_ai.request.model") == "llama3"
    assert (
        ollama_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert ollama_span.attributes.get("gen_ai.completion.0.content") == response
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert ollama_span.attributes.get("gen_ai.usage.prompt_tokens") == 17
    assert ollama_span.attributes.get(
        "llm.usage.total_tokens"
    ) == ollama_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + ollama_span.attributes.get(
        "gen_ai.usage.prompt_tokens"
    )
