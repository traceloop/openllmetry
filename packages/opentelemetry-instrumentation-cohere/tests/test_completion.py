import os
import pytest
import cohere


@pytest.mark.vcr
def test_cohere_completion(exporter):
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))
    res = co.generate(model="command", prompt="Tell me a joke, pirate style")

    spans = exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.completion"
    assert cohere_span.attributes.get("gen_ai.system") == "Cohere"
    assert cohere_span.attributes.get("llm.request.type") == "completion"
    assert cohere_span.attributes.get("gen_ai.request.model") == "command"
    assert (
        cohere_span.attributes.get("gen_ai.completion.0.content")
        == res.generations[0].text
    )
