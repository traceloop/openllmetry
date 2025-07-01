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
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

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


@pytest.mark.vcr
def test_titan_invoke_stream(test_context, brt):
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

    response = brt.invoke_model_with_response_stream(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    stream = response.get("body")
    response_body = None
    generated_text = []
    if stream:
        for event in stream:
            if "chunk" in event:
                response_body = json.loads(event["chunk"].get("bytes").decode())
                generated_text.append(response_body["outputText"])

    assert len(generated_text) > 0
    # response_body = json.loads(response.get("body").read())

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
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

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
    completion_text = "".join(generated_text)
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == completion_text
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.5
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None


@pytest.mark.vcr
def test_titan_converse(test_context, brt):
    guardrail = {
        "guardrailIdentifier": "5zwrmdlsra2e",
        "guardrailVersion": "DRAFT",
        "trace": "enabled",
    }
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "guardContent": {
                        "text": {
                            "text": "Tokyo is the capital of Japan.",
                            "qualifiers": ["grounding_source"],
                        }
                    }
                },
                {
                    "guardContent": {
                        "text": {
                            "text": "What is the capital of Japan?",
                            "qualifiers": ["query"],
                        }
                    }
                },
            ],
        }
    ]

    modelId = "amazon.titan-text-express-v1"

    response = brt.converse(
        modelId=modelId,
        messages=messages,
        guardrailConfig=guardrail,
    )

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert bedrock_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    ] == json.dumps(messages[0].get("content"), default=str)

    # Assert on response
    generated_text = response["output"]["message"]["content"]
    for i in range(0, len(generated_text)):
        assert (
            bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content"]
            == generated_text[i]["text"]
        )


@pytest.mark.vcr
def test_titan_converse_stream(test_context, brt):
    guardrail = {
        "guardrailIdentifier": "5zwrmdlsra2e",
        "guardrailVersion": "DRAFT",
        "trace": "enabled",
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "guardContent": {
                        "text": {
                            "text": "Tokyo is the capital of Japan.",
                            "qualifiers": ["grounding_source"],
                        }
                    }
                },
                {
                    "guardContent": {
                        "text": {
                            "text": "What is the capital of Japan?",
                            "qualifiers": ["query"],
                        }
                    }
                },
            ],
        }
    ]

    modelId = "amazon.titan-text-express-v1"

    response = brt.converse_stream(
        modelId=modelId,
        messages=messages,
        guardrailConfig=guardrail,
    )

    stream = response.get("stream")

    response_role = None
    content = ""
    inputTokens = 0
    outputTokens = 0

    if stream:
        for event in stream:

            if "messageStart" in event:
                response_role = event["messageStart"]["role"]

            if "contentBlockDelta" in event:
                content += event["contentBlockDelta"]["delta"]["text"]

            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    inputTokens = metadata["usage"]["inputTokens"]
                    outputTokens = metadata["usage"]["outputTokens"]

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

        # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

    # Assert on request type  
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert bedrock_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    ] == json.dumps(messages[0].get("content"), default=str)

    # Assert on response
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == content
    )
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"]
        == response_role
    )

    # Assert on usage data
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == inputTokens
    )
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        == outputTokens
    )
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == inputTokens + outputTokens
    )
