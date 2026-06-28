"""Embedding tracing tests (offline via litellm mock_response)."""

import litellm
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

MOCK_EMBEDDING = [0.1, 0.2, 0.3, 0.4]
_EMBEDDINGS = GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value


def test_embedding(instrument_legacy, span_exporter):
    litellm.embedding(
        model="text-embedding-ada-002",
        input="The quick brown fox",
        mock_response=[MOCK_EMBEDDING],
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.embeddings"]
    attrs = spans[0].attributes
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == _EMBEDDINGS
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "text-embedding-ada-002"


@pytest.mark.asyncio
async def test_aembedding(instrument_legacy, span_exporter):
    await litellm.aembedding(
        model="text-embedding-ada-002",
        input="The quick brown fox",
        mock_response=[MOCK_EMBEDDING],
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.embeddings"]
    assert spans[0].attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == _EMBEDDINGS
