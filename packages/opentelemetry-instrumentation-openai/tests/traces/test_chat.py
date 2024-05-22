import pytest
from opentelemetry.semconv.ai import SpanAttributes


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
        open_ai_span.attributes["gen_ai.prompt.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")
    assert (
        open_ai_span.attributes.get("openai.api_base") == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.openai.system_fingerprint")
        == "fp_2b778c6b35"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False


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
        open_ai_span.attributes["gen_ai.prompt.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")
    assert (
        open_ai_span.attributes.get("openai.api_base") == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS)
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
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
        open_ai_span.attributes["gen_ai.prompt.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")
    assert (
        open_ai_span.attributes.get("openai.api_base") == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS)
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens
