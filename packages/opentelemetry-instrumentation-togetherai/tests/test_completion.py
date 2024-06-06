import os
import pytest
from together import Together


@pytest.mark.vcr
def test_togetherai_completion(exporter):
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    response = client.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt="Tell me a joke about OpenTelemetry.",
    )

    spans = exporter.get_finished_spans()
    togetherai_span = spans[0]
    assert togetherai_span.name == "together.completion"
    assert togetherai_span.attributes.get("gen_ai.system") == "TogetherAI"
    assert togetherai_span.attributes.get("llm.request.type") == "completion"
    assert (
        togetherai_span.attributes.get("gen_ai.request.model")
        == "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    assert (
        togetherai_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry."
    )
    assert (
        togetherai_span.attributes.get("gen_ai.completion.0.content")
        == response.choices[0].text
    )
    assert togetherai_span.attributes.get("gen_ai.usage.prompt_tokens") == 10
    assert togetherai_span.attributes.get(
        "llm.usage.total_tokens"
    ) == togetherai_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + togetherai_span.attributes.get(
        "gen_ai.usage.prompt_tokens"
    )
