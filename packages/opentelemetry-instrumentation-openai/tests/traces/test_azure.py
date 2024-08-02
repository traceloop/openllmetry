import pytest
from opentelemetry.semconv_ai import SpanAttributes
import json

PROMPT_FILTER_KEY = "prompt_filter_results"
PROMPT_ERROR = "prompt_error"


@pytest.mark.vcr
def test_chat(exporter, azure_openai_client):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com//openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False


@pytest.mark.vcr
def test_chat_content_filtering(exporter, azure_openai_client):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == "FILTERED"
    )
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com//openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False

    content_filter_json = open_ai_span.attributes.get(
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content_filter_results"
    )

    assert len(content_filter_json) > 0

    content_filter_results = json.loads(content_filter_json)

    assert content_filter_results["hate"]["filtered"] is True
    assert content_filter_results["hate"]["severity"] == "high"
    assert content_filter_results["self_harm"]["filtered"] is False
    assert content_filter_results["self_harm"]["severity"] == "safe"


@pytest.mark.vcr
def test_prompt_content_filtering(exporter, azure_openai_client):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert isinstance(
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_ERROR}"], str
    )

    error = json.loads(
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_ERROR}"]
    )

    assert "innererror" in error

    assert "content_filter_result" in error["innererror"]

    assert error["innererror"]["code"] == "ResponsibleAIPolicyViolation"

    assert error["innererror"]["content_filter_result"]["hate"]["filtered"]

    assert error["innererror"]["content_filter_result"]["hate"]["severity"] == "high"

    assert error["innererror"]["content_filter_result"]["sexual"]["filtered"] is False

    assert error["innererror"]["content_filter_result"]["sexual"]["severity"] == "safe"


@pytest.mark.vcr
def test_chat_streaming(exporter, azure_openai_client):
    response = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
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
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com//openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # prompt filter results
    prompt_filter_results = json.loads(
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_FILTER_KEY}")
    )
    assert prompt_filter_results[0]["prompt_index"] == 0
    assert (
        prompt_filter_results[0]["content_filter_results"]["hate"]["severity"] == "safe"
    )
    assert (
        prompt_filter_results[0]["content_filter_results"]["self_harm"]["filtered"]
        is False
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming(exporter, async_azure_openai_client):
    response = await async_azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
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
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com//openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count
