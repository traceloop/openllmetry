import os
import pytest
import cohere


@pytest.mark.vcr
def test_cohere_chat(exporter):
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))
    res = co.chat(model="command", message="Tell me a joke, pirate style")

    spans = exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get("gen_ai.system") == "Cohere"
    assert cohere_span.attributes.get("llm.request.type") == "chat"
    assert cohere_span.attributes.get("gen_ai.request.model") == "command"
    assert (
        cohere_span.attributes.get("gen_ai.prompt.0.content")
        == "Tell me a joke, pirate style"
    )
    assert cohere_span.attributes.get("gen_ai.completion.0.content") == res.text
    assert cohere_span.attributes.get("gen_ai.usage.prompt_tokens") == 69
    assert cohere_span.attributes.get(
        "llm.usage.total_tokens"
    ) == cohere_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + cohere_span.attributes.get(
        "gen_ai.usage.prompt_tokens"
    )
