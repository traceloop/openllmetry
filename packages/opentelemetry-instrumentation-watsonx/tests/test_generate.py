import pytest
import sys


@pytest.mark.skipif(sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10")
@pytest.mark.vcr
def test_generate(exporter, watson_ai_model):
    if watson_ai_model is None:
        print("test_generate test skipped.")
        return
    watson_ai_model.generate(prompt="What is 1 + 1?")
    spans = exporter.get_finished_spans()
    watsonx_ai_span = spans[1]
    assert watsonx_ai_span.attributes["llm.prompts.user"] == "What is 1 + 1?"
    assert watsonx_ai_span.attributes["llm.vendor"] == "Watsonx"
    assert watsonx_ai_span.attributes.get("llm.completions.content")
    assert watsonx_ai_span.attributes.get("llm.usage.total_tokens")
