import os
import pytest
from mistralai.client import MistralClient
from mistralai.async_client import MistralAsyncClient
from opentelemetry.semconv.ai import SpanAttributes


@pytest.mark.vcr
def test_ollama_embeddings(exporter):
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    client.embeddings(
        model="mistral-embed",
        input="Tell me a joke about OpenTelemetry",
    )

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "mistralai.embeddings"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}") == "mistral-embed"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistral_async_embeddings(exporter):
    client = MistralAsyncClient(api_key=os.environ["MISTRAL_API_KEY"])
    await client.embeddings(
        model="mistral-embed",
        input=["Tell me a joke about OpenTelemetry", "Tell me a joke about Traceloop"],
    )

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "mistralai.embeddings"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}") == "mistral-embed"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
