import pytest
from opentelemetry.semconv_ai import SpanAttributes

import json


@pytest.mark.vcr
def test_imported_model_completion(test_context, brt):
    prompt = "Explain quantum mechanics."
    payload = {"prompt": prompt, "max_tokens": 100, "topP": 2, "temperature": 0.5}
    payload_str = json.dumps(payload)

    model_arn = "arn:aws:bedrock:us-east-1:12345:imported-model/13r66ttyg"
    response = brt.invoke_model(modelId=model_arn, body=payload_str)
    data = json.loads(response["body"].read().decode("utf-8"))
    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"
    imported_model_span = spans[0]

    assert (
        imported_model_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "arn:aws:bedrock:us-east-1:12345:imported-model/13r66ttyg"
    )
    assert (
        imported_model_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    )
    assert imported_model_span.attributes[SpanAttributes.LLM_SYSTEM] == "imported_model"
    assert imported_model_span.attributes.get("gen_ai.response.id") is None
    assert imported_model_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 100
    assert imported_model_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.5
    assert imported_model_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 2

    assert (
        imported_model_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == prompt
    )
    assert (
        imported_model_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == data["generation"]
    )
