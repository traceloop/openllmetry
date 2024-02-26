import pytest


@pytest.mark.vcr
def disabled_test_generate(exporter, watson_ai_model):
    watson_ai_model.generate(prompt="What is 1 + 1?")
    spans = exporter.get_finished_spans()
    watsonx_ai_span = spans[0]
    assert watsonx_ai_span.attributes["llm.prompts.user"] == "What is 1 + 1?"
    assert watsonx_ai_span.attributes["llm.vendor"] == "Watsonx"
    assert watsonx_ai_span.attributes.get("llm.completions.content")
    assert watsonx_ai_span.attributes.get("llm.usage.total_tokens")


# Remove once the above test is re-enabled
def test_noop():
    pass
