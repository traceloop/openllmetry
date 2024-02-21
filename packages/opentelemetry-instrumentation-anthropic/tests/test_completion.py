import pytest
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


@pytest.mark.vcr
def test_anthropic_completion(exporter):
    client = Anthropic()
    client.completions.create(
        prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.completion",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes["llm.prompts.0.user"]
        == f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    )
    assert anthropic_span.attributes.get("llm.completions.0.content")
