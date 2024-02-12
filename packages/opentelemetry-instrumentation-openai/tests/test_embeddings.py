import pytest


@pytest.mark.vcr
def test_embeddings(exporter, openai_client):
    openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["llm.prompts.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes["llm.request.model"] == "text-embedding-ada-002"
    assert open_ai_span.attributes["llm.usage.prompt_tokens"] == 8
