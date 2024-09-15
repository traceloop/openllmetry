import pytest
from opentelemetry.semconv_ai import SpanAttributes

import json


@pytest.mark.vcr()
def test_titan_completion(exporter, brt):
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

    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    bedrock_span = spans[0]
    assert bedrock_span.attributes[
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    ] == response_body.get("inputTextTokenCount")

    assert bedrock_span.attributes[
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ] == response_body.get("results")[0].get("tokenCount")

    assert (
        bedrock_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == bedrock_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        + bedrock_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
    )
