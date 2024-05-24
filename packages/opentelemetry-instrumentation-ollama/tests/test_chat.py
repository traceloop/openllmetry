import pytest
import ollama


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
    assert ollama_span.attributes.get("gen_ai.system") == "Ollama"
    assert ollama_span.attributes.get("llm.request.type") == "chat"
    assert not ollama_span.attributes.get("llm.is_streaming")
    assert ollama_span.attributes.get("gen_ai.request.model") == "llama3"
    assert (
        ollama_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get("gen_ai.completion.0.content")
        == response["message"]["content"]
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
    assert ollama_span.attributes.get("gen_ai.system") == "Ollama"
    assert ollama_span.attributes.get("llm.request.type") == "chat"
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
    assert ollama_span.attributes.get("gen_ai.system") == "Ollama"
    assert ollama_span.attributes.get("llm.request.type") == "chat"
    assert not ollama_span.attributes.get("llm.is_streaming")
    assert ollama_span.attributes.get("gen_ai.request.model") == "llama3"
    assert (
        ollama_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get("gen_ai.completion.0.content")
        == response["message"]["content"]
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
    assert ollama_span.attributes.get("gen_ai.system") == "Ollama"
    assert ollama_span.attributes.get("llm.request.type") == "chat"
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
