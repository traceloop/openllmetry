import pytest
import json
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_cohere_completion(test_context, brt):
    prompt = "Tell me a joke about opentelemetry"
    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.5,
            "p": 0.5,
        }
    )

    modelId = "cohere.command-text-v14"
    accept = "application/json"
    contentType = "application/json"

    response = brt.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    response_body = json.loads(response.get("body").read())

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "command-text-v14"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "cohere"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"] == prompt

    # Assert on response
    generated_text = response_body["generations"][0]["text"]
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == generated_text
    )
    assert (
        bedrock_span.attributes.get("gen_ai.response.id")
        == "3266ca30-473c-4491-b6ef-5b1f033798d2"
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.5
