import pytest
from opentelemetry.semconv_ai import SpanAttributes
from pydantic import BaseModel


class StructuredAnswer(BaseModel):
    rating: int
    joke: str


@pytest.mark.vcr
def test_parsed_completion(exporter, openai_client):
    openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
            open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
            == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
            open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
            == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-AGC1gNoe1Zyq9yZicdhLc85lmt2Ep"


@pytest.mark.vcr
def test_parsed_refused_completion(exporter, openai_client):
    openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in open_ai_span.attributes
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.refusal" in open_ai_span.attributes
    assert (
            open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.refusal"]
            == "I'm very sorry, but I can't assist with that request."
    )
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-AGky8KFDbg6f5fF4qLtsBredIjZZh"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_completion(exporter, async_openai_client):
    await async_openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
            open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
            == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
            open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
            == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-AGC1iysV7rZ0qZ510vbeKVTNxSOHB"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_refused_completion(exporter, async_openai_client):
    await async_openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in open_ai_span.attributes
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.refusal" in open_ai_span.attributes
    assert (
            open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.refusal"]
            == "I'm very sorry, but I can't assist with that request."
    )
    assert open_ai_span.attributes.get("gen_ai.response.id") == "chatcmpl-AGkyFJGzZPUGAAEDJJuOS3idKvD3G"
