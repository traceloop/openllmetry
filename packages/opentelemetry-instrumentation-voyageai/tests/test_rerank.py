import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_voyageai_rerank_legacy(
    span_exporter, instrument_legacy, voyageai_client
):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France and is known for the Eiffel Tower.",
        "London is the capital of England and has Big Ben.",
        "Berlin is the capital of Germany.",
        "The capital of France is Paris, a beautiful city.",
    ]

    result = voyageai_client.rerank(
        query=query,
        documents=documents,
        model="rerank-2-lite",
        top_k=3,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    voyageai_span = spans[0]
    assert voyageai_span.name == "voyageai.rerank"
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "voyageai"
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "rerank"
    assert (
        voyageai_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "rerank-2-lite"
    )

    # Check that documents are captured as system prompts
    for i, doc in enumerate(documents):
        assert (
            voyageai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.{i}.role")
            == "system"
        )
        assert (
            voyageai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.{i}.content")
            == doc
        )

    # Check that query is captured as user prompt
    assert (
        voyageai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_PROMPT}.{len(documents)}.role"
        )
        == "user"
    )
    assert (
        voyageai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_PROMPT}.{len(documents)}.content"
        )
        == query
    )

    # Check token usage is captured
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) is not None

    # Check that results are captured as completions
    for idx in range(len(result.results)):
        assert (
            voyageai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}.role")
            == "assistant"
        )
        content = voyageai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}.content")
        assert content is not None
        assert "Doc" in content
        assert "Score:" in content

    # Verify results were returned
    assert result is not None
    assert len(result.results) == 3


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_voyageai_rerank_async_legacy(
    span_exporter, instrument_legacy, async_voyageai_client
):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France and is known for the Eiffel Tower.",
        "London is the capital of England and has Big Ben.",
        "Berlin is the capital of Germany.",
    ]

    result = await async_voyageai_client.rerank(
        query=query,
        documents=documents,
        model="rerank-2-lite",
        top_k=2,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    voyageai_span = spans[0]
    assert voyageai_span.name == "voyageai.rerank"
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "voyageai"
    assert voyageai_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "rerank"
    assert (
        voyageai_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "rerank-2-lite"
    )

    # Verify results were returned
    assert result is not None
    assert len(result.results) == 2
