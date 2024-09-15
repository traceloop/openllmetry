import sys

import pytest
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10"
)
@pytest.mark.vcr
def test_generate(exporter, watson_ai_model):
    if watson_ai_model is None:
        print("test_generate test skipped.")
        return
    watson_ai_model.generate(prompt="What is 1 + 1?")
    spans = exporter.get_finished_spans()
    watsonx_ai_span = spans[1]
    assert (
        watsonx_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == "What is 1 + 1?"
    )
    assert watsonx_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "Watsonx"
    assert watsonx_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert watsonx_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10"
)
@pytest.mark.vcr
def test_generate_text_stream(exporter, watson_ai_model):
    if watson_ai_model is None:
        print("test_generate_text_stream test skipped.")
        return
    response = watson_ai_model.generate_text_stream(
        prompt="Write an epigram about the sun"
    )
    generated_text = ""
    for chunk in response:
        generated_text += chunk
    spans = exporter.get_finished_spans()
    watsonx_ai_span = spans[1]
    assert (
        watsonx_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == "Write an epigram about the sun"
    )
    assert watsonx_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "Watsonx"
    assert (
        watsonx_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == generated_text
    )
    assert watsonx_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
