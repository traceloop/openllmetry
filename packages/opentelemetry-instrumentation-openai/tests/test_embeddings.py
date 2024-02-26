import openai
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
    assert open_ai_span.attributes["openai.api_base"] == "https://api.openai.com/v1/"


@pytest.mark.vcr
def test_azure_openai_embeddings(exporter):
    api_key = "redacted"
    azure_deployment = "redacted"

    openai_client = openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=f"https://{azure_deployment}.openai.azure.com",
        api_version="2023-07-01-preview",
    )
    openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="embedding",
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
    assert open_ai_span.attributes["llm.request.model"] == "embedding"
    assert open_ai_span.attributes["llm.usage.prompt_tokens"] == 8
    assert open_ai_span.attributes["openai.api_base"] == "https://redacted.openai.azure.com/openai/"
    assert open_ai_span.attributes["openai.api_version"] == "2023-07-01-preview"
