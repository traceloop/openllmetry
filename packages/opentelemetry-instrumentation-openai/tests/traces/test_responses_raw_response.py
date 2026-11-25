import pytest
from openai import AsyncOpenAI, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr
def test_responses_with_raw_response(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    raw = openai_client.responses.with_raw_response.create(
        model="gpt-6-luna",
        input="What is the capital of France?",
        max_output_tokens=64,
    )
    assert raw.http_response.status_code == 200
    response = raw.parse()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-6-luna"
    assert span.attributes["gen_ai.response.model"] == response.model
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.usage.input_tokens"] == response.usage.input_tokens
    assert span.attributes["gen_ai.usage.output_tokens"] == response.usage.output_tokens


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_responses_with_raw_response(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    raw = await async_openai_client.responses.with_raw_response.create(
        model="gpt-6-luna",
        input="What is the capital of France?",
        max_output_tokens=64,
    )
    assert raw.http_response.status_code == 200
    response = raw.parse()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-6-luna"
    assert span.attributes["gen_ai.response.model"] == response.model
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.usage.input_tokens"] == response.usage.input_tokens
    assert span.attributes["gen_ai.usage.output_tokens"] == response.usage.output_tokens
