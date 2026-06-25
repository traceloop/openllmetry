"""Provider (``gen_ai.system``) resolution tests.

LiteLLM uses provider prefixes (``azure``, ``bedrock``, ``vertex_ai``, ...) that differ
from the published ``gen_ai.system`` values in opentelemetry-semantic-conventions-ai.
These assert the instrumentor normalizes them so litellm spans share the same system
key as the rest of the repo, while unknown providers pass through unchanged.
"""

import pytest
from opentelemetry.instrumentation.litellm import _resolve_system_from_kwargs
from opentelemetry.semconv_ai import GenAISystem


@pytest.mark.parametrize(
    "model,expected",
    [
        ("azure/gpt-4o", GenAISystem.AZURE.value),
        ("bedrock/anthropic.claude-3", GenAISystem.AWS.value),
        ("vertex_ai/gemini-1.5-pro", GenAISystem.GOOGLE.value),
        ("gemini/gemini-1.5-pro", GenAISystem.GOOGLE.value),
        ("mistral/mistral-large", GenAISystem.MISTRALAI.value),
        # OpenAI already matches the published value; unknown providers pass through.
        ("openai/gpt-4o", "openai"),
        ("someprovider/some-model", "someprovider"),
    ],
)
def test_system_resolution_normalizes_aliases(model, expected):
    assert _resolve_system_from_kwargs({"model": model}) == expected


def test_explicit_custom_llm_provider_is_normalized():
    assert (
        _resolve_system_from_kwargs({"model": "x", "custom_llm_provider": "bedrock"})
        == GenAISystem.AWS.value
    )


def test_bare_model_falls_back_to_litellm():
    assert _resolve_system_from_kwargs({"model": "gpt-3.5-turbo"}) == "litellm"
