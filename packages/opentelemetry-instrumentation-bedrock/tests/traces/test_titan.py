import pytest
from opentelemetry.semconv_ai import SpanAttributes

import json


@pytest.mark.vcr
def test_titan_completion(test_context, brt):
    body = json.dumps(
        {
            "inputText": "Translate to spanish: 'Amazon Bedrock is the easiest way to build and"
            + "scale generative AI applications with base models (FMs)'.",
            "textGenerationConfig": {
                "maxTokenCount": 200,
                "temperature": 0.5,
                "topP": 0.5,
            },
        }
    )

    modelId = "amazon.titan-text-express-v1"
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
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "amazon"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on prompt
    expected_prompt = (
        "Translate to spanish: 'Amazon Bedrock is the easiest way to build and"
        "scale generative AI applications with base models (FMs)'."
    )
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == expected_prompt
    )

    # Assert on response
    generated_text = response_body["results"][0]["outputText"]
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == generated_text
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.5
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None
