import json
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_voyageai_embed_legacy(
    span_exporter, instrument_legacy, voyageai_client
):
    texts = [
        "The capital of France is Paris.",
        "London is the capital of England.",
        "Berlin is the capital of Germany.",
    ]

    result = voyageai_client.embed(
        texts=texts,
        model="voyage-3-lite",
        input_type="document",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    voyageai_span = spans[0]
    assert voyageai_span.name == "voyageai.embed"
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "voyageai"
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "embeddings"
    assert (
        voyageai_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "voyage-3-lite"
    )
    assert (
        voyageai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"
        )
        == "user"
    )

    # Check that input texts are captured
    content = voyageai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
    assert content is not None
    parsed_content = json.loads(content)
    assert len(parsed_content) == 3
    assert parsed_content[0]["text"] == texts[0]

    # Check token usage is captured
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) is not None

    # Check embedding dimension is captured
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT) is not None

    # Verify embeddings were returned
    assert result is not None
    assert len(result.embeddings) == 3


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_voyageai_embed_async_legacy(
    span_exporter, instrument_legacy, async_voyageai_client
):
    texts = [
        "The capital of France is Paris.",
        "London is the capital of England.",
    ]

    result = await async_voyageai_client.embed(
        texts=texts,
        model="voyage-3-lite",
        input_type="document",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    voyageai_span = spans[0]
    assert voyageai_span.name == "voyageai.embed"
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "voyageai"
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "embeddings"
    assert (
        voyageai_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "voyage-3-lite"
    )

    # Verify embeddings were returned
    assert result is not None
    assert len(result.embeddings) == 2
