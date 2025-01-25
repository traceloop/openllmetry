import os
import pytest
from mistralai.client import MistralClient
from mistralai.async_client import MistralAsyncClient
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_mistral_embeddings(exporter):
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    client.embeddings(
        model="mistral-embed",
        input="Tell me a joke about OpenTelemetry",
    )

    spans = exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.embeddings"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-embed"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert mistral_span.attributes.get("gen_ai.response.id") == "3fe947e29a95441a94086e11de21bff1"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistral_async_embeddings(exporter):
    client = MistralAsyncClient(api_key=os.environ["MISTRAL_API_KEY"])
    await client.embeddings(
        model="mistral-embed",
        input=["Tell me a joke about OpenTelemetry", "Tell me a joke about Traceloop"],
    )

    spans = exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.embeddings"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-embed"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert mistral_span.attributes.get("gen_ai.response.id") == "220426da5cd84a8391a0d65738c90dc8"
