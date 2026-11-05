"""Regression tests: span attributes must be captured for positional arguments.

The Voyage AI docs call ``embed(texts, model=...)`` and
``rerank(query, documents, model)`` positionally, so the instrumentation must
not rely on ``kwargs`` alone. These tests stub the network layer, so they need
neither an API key nor a VCR cassette.
"""

import json
from types import SimpleNamespace

import pytest
import voyageai
from opentelemetry.instrumentation.voyageai import VoyageAIInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

TEXTS = ["Paris is in France.", "Berlin is in Germany."]
QUERY = "What is the capital of France?"
DOCUMENTS = ["Paris", "Berlin", "Madrid"]


@pytest.fixture
def stubbed_instrumented_client(tracer_provider, monkeypatch):
    def fake_embed(self, texts, model=None, input_type=None, **kwargs):
        return SimpleNamespace(total_tokens=7, embeddings=[[0.1, 0.2]] * len(texts))

    def fake_rerank(self, query, documents, model, top_k=None, **kwargs):
        return SimpleNamespace(total_tokens=7, results=[])

    monkeypatch.setattr(voyageai.Client, "embed", fake_embed)
    monkeypatch.setattr(voyageai.Client, "rerank", fake_rerank)

    instrumentor = VoyageAIInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield voyageai.Client(api_key="test_api_key")
    instrumentor.uninstrument()


def test_embed_positional_args(span_exporter, stubbed_instrumented_client):
    stubbed_instrumented_client.embed(TEXTS, "voyage-3-lite", input_type="document")

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "voyage-3-lite"
    assert span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    content = json.loads(span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"])
    assert [item["text"] for item in content] == TEXTS


def test_rerank_positional_args(span_exporter, stubbed_instrumented_client):
    stubbed_instrumented_client.rerank(QUERY, DOCUMENTS, "rerank-2-lite", top_k=2)

    (span,) = span_exporter.get_finished_spans()
    prefix = GenAIAttributes.GEN_AI_PROMPT
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "rerank-2-lite"
    for index, document in enumerate(DOCUMENTS):
        assert span.attributes[f"{prefix}.{index}.content"] == document
    assert span.attributes[f"{prefix}.{len(DOCUMENTS)}.role"] == "user"
    assert span.attributes[f"{prefix}.{len(DOCUMENTS)}.content"] == QUERY
