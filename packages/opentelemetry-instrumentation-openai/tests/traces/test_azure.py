import pytest
from opentelemetry.semconv.ai import SpanAttributes


@pytest.mark.vcr
def test_chat(exporter, azure_openai_client):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["gen_ai.prompt.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")
    assert (
        open_ai_span.attributes.get("openai.api_base")
        == "https://traceloop-stg.openai.azure.com//openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False


@pytest.mark.vcr
def test_chat_content_filtering(exporter, azure_openai_client):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["gen_ai.prompt.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content") == "FILTERED"
    assert (
        open_ai_span.attributes.get("openai.api_base")
        == "https://traceloop-stg.openai.azure.com//openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False


@pytest.mark.vcr
def test_chat_streaming(exporter, azure_openai_client):
    response = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
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
        open_ai_span.attributes["gen_ai.prompt.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")
    assert (
        open_ai_span.attributes.get("openai.api_base")
        == "https://traceloop-stg.openai.azure.com//openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming(exporter, async_azure_openai_client):
    response = await async_azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
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
    print("HEYY", open_ai_span.attributes)
    assert (
        open_ai_span.attributes["gen_ai.prompt.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")
    assert (
        open_ai_span.attributes.get("openai.api_base")
        == "https://traceloop-stg.openai.azure.com//openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count
