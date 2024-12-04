import httpx
import openai
import pytest
from unittest.mock import patch
from opentelemetry.semconv_ai import SpanAttributes
from .utils import spy_decorator, assert_request_contains_tracecontext


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
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "text-embedding-ada-002"
    )
    assert open_ai_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )


@pytest.mark.vcr
def test_embeddings_with_raw_response(exporter, openai_client):
    response = openai_client.embeddings.with_raw_response.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )

    assert (
        open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "text-embedding-ada-002"
    )
    assert open_ai_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )

    parsed_response = response.parse()
    assert parsed_response.data[0]


@pytest.mark.vcr
def test_azure_openai_embeddings(exporter):
    api_key = "test-api-key"
    azure_resource = "test-resource"
    azure_deployment = "test-deployment"

    openai_client = openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=f"https://{azure_resource}.openai.azure.com",
        azure_deployment=azure_deployment,
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
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "embedding"
    assert open_ai_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == f"https://{azure_resource}.openai.azure.com/openai/deployments/{azure_deployment}/"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_VERSION]
        == "2023-07-01-preview"
    )


@pytest.mark.vcr
def test_embeddings_context_propagation(exporter, vllm_openai_client):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_embeddings_context_propagation(exporter, async_vllm_openai_client):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)
