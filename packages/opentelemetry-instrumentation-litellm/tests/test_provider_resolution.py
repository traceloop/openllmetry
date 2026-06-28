"""Provider (``gen_ai.provider.name``) resolution tests.

LiteLLM uses provider prefixes (``azure``, ``bedrock``, ``vertex_ai``, ...) that differ
from the OTel ``gen_ai.provider.name`` well-known values. These assert the instrumentor
normalizes them to the upstream values, while unknown providers pass through unchanged.
"""

import pytest
from opentelemetry.instrumentation.litellm import _resolve_provider_from_kwargs
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

_Provider = GenAIAttributes.GenAiProviderNameValues


@pytest.mark.parametrize(
    "model,expected",
    [
        ("azure/gpt-4o", _Provider.AZURE_AI_OPENAI.value),
        ("azure_ai/grok-3", _Provider.AZURE_AI_INFERENCE.value),
        ("bedrock/anthropic.claude-3", _Provider.AWS_BEDROCK.value),
        ("vertex_ai/gemini-1.5-pro", _Provider.GCP_VERTEX_AI.value),
        ("gemini/gemini-1.5-pro", _Provider.GCP_GEMINI.value),
        ("mistral/mistral-large", _Provider.MISTRAL_AI.value),
        # OpenAI already matches the published value; unknown providers pass through.
        ("openai/gpt-4o", "openai"),
        ("someprovider/some-model", "someprovider"),
    ],
)
def test_provider_resolution_normalizes_aliases(model, expected):
    assert _resolve_provider_from_kwargs({"model": model}) == expected


def test_explicit_custom_llm_provider_is_normalized():
    assert (
        _resolve_provider_from_kwargs({"model": "x", "custom_llm_provider": "bedrock"})
        == _Provider.AWS_BEDROCK.value
    )


def test_bare_model_falls_back_to_litellm():
    assert _resolve_provider_from_kwargs({"model": "gpt-3.5-turbo"}) == "litellm"
