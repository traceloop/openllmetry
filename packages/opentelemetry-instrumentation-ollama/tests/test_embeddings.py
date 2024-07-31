import pytest
import ollama
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_ollama_embeddings(exporter):
    ollama.embeddings(model="llama3", prompt="Tell me a joke about OpenTelemetry")

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.embeddings"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Ollama"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
