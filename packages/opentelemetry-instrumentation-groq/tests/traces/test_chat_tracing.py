import pytest
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_chat(exporter, groq_client):
    groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert (
        groq_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert groq_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat(exporter, async_groq_client):
    await async_groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert (
        groq_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert groq_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
