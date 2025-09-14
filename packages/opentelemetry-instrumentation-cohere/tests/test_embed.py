import json
import pytest
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_cohere_v2_embed_legacy(
    span_exporter, log_exporter, instrument_legacy, cohere_client_v2
):
    texts = [
        "Carson City is the capital city of the American state of Nevada."
        + " At the  2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands"
        + " in the Pacific Ocean that are a political division controlled by the "
        + "United States. Its capital is Saipan.",
        "Charlotte Amalie is the capital and largest city of the United States "
        + "Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially "
        + "as the District of Columbia) is the capital of the United States. It is a federal district. ",
        "Capital punishment (the death penalty) has existed in the United States "
        + "since before the United States was a country. As of 2017, capital "
        + "punishment is legal in 30 of the 50 states.",
        "North Dakota is a state in the United States. 672,591 people lived"
        + " in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
    ]

    cohere_client_v2.embed(
        input_type="search_document",
        texts=texts, model="embed-english-light-v3.0"
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.embed"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "embedding"
    assert (
        cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL)
        == "embed-english-light-v3.0"
    )
    assert (
        cohere_span.attributes.get(
            f"{SpanAttributes.LLM_PROMPTS}.0.role"
        )
        == "user"
    )
    assert json.loads(cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")) == [{
        "type": "text",
        "text": text
    } for text in texts]

    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "74d7aae4-2939-4002-b4d4-352dfcca03cf"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_v2_embed_legacy_async(
    span_exporter, log_exporter, instrument_legacy, async_cohere_client_v2
):
    texts = [
        "Carson City is the capital city of the American state of Nevada."
        + " At the  2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands"
        + " in the Pacific Ocean that are a political division controlled by the "
        + "United States. Its capital is Saipan.",
        "Charlotte Amalie is the capital and largest city of the United States "
        + "Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially "
        + "as the District of Columbia) is the capital of the United States. It is a federal district. ",
        "Capital punishment (the death penalty) has existed in the United States "
        + "since before the United States was a country. As of 2017, capital "
        + "punishment is legal in 30 of the 50 states.",
        "North Dakota is a state in the United States. 672,591 people lived"
        + " in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
    ]

    await async_cohere_client_v2.embed(
        input_type="search_document",
        texts=texts, model="embed-english-light-v3.0"
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.embed"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "embedding"
    assert (
        cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL)
        == "embed-english-light-v3.0"
    )
    assert (
        cohere_span.attributes.get(
            f"{SpanAttributes.LLM_PROMPTS}.0.role"
        )
        == "user"
    )
    assert json.loads(cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")) == [{
        "type": "text",
        "text": text
    } for text in texts]

    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "3bc87f83-0534-4478-af7d-fee196c92758"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"
