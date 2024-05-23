import os

import cohere
import pytest
from opentelemetry.semconv.ai import SpanAttributes


@pytest.mark.vcr
def test_cohere_chat(exporter):
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))
    res = co.chat(model="command", message="Tell me a joke, pirate style")

    spans = exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke, pirate style"
    )
    assert cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == res.text
    assert cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 58
    assert cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )
