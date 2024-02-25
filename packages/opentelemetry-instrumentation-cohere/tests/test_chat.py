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
    assert cohere_span.attributes.get("llm.vendor") == "Cohere"
    assert cohere_span.attributes.get("llm.request.type") == "chat"
    assert cohere_span.attributes.get("llm.request.model") == "command"
    assert (
        cohere_span.attributes.get("llm.prompts.0.content")
        == "Tell me a joke, pirate style"
    )
    assert cohere_span.attributes.get("llm.completions.0.content") == res.text
    assert cohere_span.attributes.get("llm.usage.prompt_tokens") == 69
    assert cohere_span.attributes.get(
        "llm.usage.total_tokens"
    ) == cohere_span.attributes.get(
        "llm.usage.completion_tokens"
    ) + cohere_span.attributes.get(
        "llm.usage.prompt_tokens"
    )
