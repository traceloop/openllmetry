import pytest
from opentelemetry.semconv_ai import SpanAttributes

import json


@pytest.mark.vcr
def test_ai21_j2_completion_string_content(exporter, brt):
    body = json.dumps(
        {
            "prompt": "Translate to spanish: 'Amazon Bedrock is the easiest way to build and"
            + "scale generative AI applications with base models (FMs)'.",
            "maxTokens": 200,
            "temperature": 0.5,
            "topP": 0.5,
        }
    )

    modelId = "ai21.j2-mid-v1"
    accept = "application/json"
    contentType = "application/json"

    response = brt.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    response_body = json.loads(response.get("body").read())

    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert meta_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == len(
        response_body.get("prompt").get("tokens")
    )
    assert meta_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == len(
        response_body.get("completions")[0].get("data").get("tokens")
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == meta_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        + meta_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
    )
