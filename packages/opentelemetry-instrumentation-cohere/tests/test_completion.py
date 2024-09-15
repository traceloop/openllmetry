import os

import cohere
import pytest
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_cohere_completion(exporter):
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))
    res = co.generate(model="command", prompt="Tell me a joke, pirate style")

    spans = exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.completion"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "completion"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == res.generations[0].text
    )
