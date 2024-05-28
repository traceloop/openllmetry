import pytest
import ollama


@pytest.mark.vcr
def test_ollama_embeddings(exporter):
    ollama.embeddings(model="llama3", prompt="Tell me a joke about OpenTelemetry")

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.embeddings"
    assert ollama_span.attributes.get("gen_ai.system") == "Ollama"
    assert ollama_span.attributes.get("llm.request.type") == "embedding"
    assert not ollama_span.attributes.get("llm.is_streaming")
    assert ollama_span.attributes.get("gen_ai.request.model") == "llama3"
    assert (
        ollama_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
