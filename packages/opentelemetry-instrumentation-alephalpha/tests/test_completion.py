import os
import pytest
from aleph_alpha_client import Client, Prompt, CompletionRequest


@pytest.mark.vcr
def test_alephalpha_completion(exporter):
    client = Client(token=os.environ.get("AA_TOKEN"))
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
    }
    request = CompletionRequest(**params)
    response = client.complete(request, model="luminous-base")

    spans = exporter.get_finished_spans()
    together_span = spans[0]
    assert together_span.name == "alephalpha.completion"
    assert together_span.attributes.get("gen_ai.system") == "AlephAlpha"
    assert together_span.attributes.get("llm.request.type") == "completion"
    assert together_span.attributes.get("gen_ai.request.model") == "luminous-base"
    assert (
        together_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke about OpenTelemetry."
    )
    assert (
        together_span.attributes.get("gen_ai.completion.0.content")
        == response.completions[0].completion
    )
    assert together_span.attributes.get("gen_ai.usage.prompt_tokens") == 9
    assert together_span.attributes.get(
        "llm.usage.total_tokens"
    ) == together_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + together_span.attributes.get(
        "gen_ai.usage.prompt_tokens"
    )
