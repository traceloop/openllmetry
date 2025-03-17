import pytest
from opentelemetry.semconv_ai import SpanAttributes

import json


@pytest.mark.vcr
def test_nova_completion(test_context, brt):

    system_list = [{"text": "tell me a very two sentence story."}]
    message_list = [{"role": "user", "content": [{"text": "A camping trip"}]}]
    inf_params = {"maxTokens": 500, "topP": 0.9, "topK": 20, "temperature": 0.7}
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }

    modelId = "amazon.nova-lite-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = brt.invoke_model(
        body=json.dumps(request_body),
        modelId=modelId,
        accept=accept,
        contentType=contentType,
    )

    response_body = json.loads(response.get("body").read())

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "amazon"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on system prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert bedrock_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    ] == system_list[0].get("text")

    # Assert on prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"

    assert bedrock_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.1.content"
    ] == json.dumps(message_list[0].get("content"), default=str)

    # Assert on response
    generated_text = response_body["output"]["message"]["content"]
    for i in range(0, len(generated_text)):
        assert (
            bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content"]
            == generated_text[i]["text"]
        )

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 500
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.7
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.9
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None


@pytest.mark.vcr
def test_nova_invoke_stream(test_context, brt):
    system_list = [{"text": "tell me a very two sentence story."}]
    message_list = [{"role": "user", "content": [{"text": "A camping trip"}]}]
    inf_params = {"maxTokens": 500, "topP": 0.9, "topK": 20, "temperature": 0.7}
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }

    modelId = "amazon.nova-lite-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = brt.invoke_model_with_response_stream(
        body=json.dumps(request_body),
        modelId=modelId,
        accept=accept,
        contentType=contentType,
    )

    stream = response.get("body")
    generated_text = []
    if stream:
        for event in stream:
            if "chunk" in event:
                response_body = json.loads(event["chunk"].get("bytes").decode())
                assert response_body is not None
                content_block_delta = response_body.get("contentBlockDelta")
                if content_block_delta:
                    generated_text.append(content_block_delta.get("delta").get("text"))

    assert len(generated_text) > 0

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "amazon"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on system prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert bedrock_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    ] == system_list[0].get("text")

    # Assert on prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"

    assert bedrock_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.1.content"
    ] == json.dumps(message_list[0].get("content"), default=str)

    # Assert on response
    completion_msg = "".join(generated_text)
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == completion_msg
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[
        SpanAttributes.LLM_REQUEST_MAX_TOKENS
    ] == inf_params.get("maxTokens")
    assert bedrock_span.attributes[
        SpanAttributes.LLM_REQUEST_TEMPERATURE
    ] == inf_params.get("temperature")
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == inf_params.get(
        "topP"
    )
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None


@pytest.mark.vcr
def test_nova_converse(test_context, brt):
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
                            "text": "Tokyo is the capital of Japan." +
                                    "The Greater Tokyo area is the most populous metropolitan area in the world.",
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
                {"text": "What is the capital of Japan?"},
            ],
        }
    ]

    system = [{"text": "You are a helpful assistant"}]

    modelId = "amazon.nova-lite-v1:0"

    inf_params = {"maxTokens": 300, "topP": 0.1, "temperature": 0.3}

    response = brt.converse(
        modelId=modelId,
        messages=messages,
        guardrailConfig=guardrail,
        system=system,
        inferenceConfig=inf_params,
    )

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "amazon"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on system prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == system[
        0
    ].get("text")

    # Assert on prompt
    assert bedrock_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.1.content"
    ] == json.dumps(messages[0].get("content"), default=str)

    # Assert on response
    generated_text = response["output"]["message"]["content"]
    for i in range(0, len(generated_text)):
        assert (
            bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content"]
            == generated_text[i]["text"]
        )

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 300
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.3
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.1


@pytest.mark.vcr
def test_nova_converse_stream(test_context, brt):
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
                            "text": "Tokyo is the capital of Japan." +
                                    "The Greater Tokyo area is the most populous metropolitan area in the world.",
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
                {"text": "What is the capital of Japan?"},
            ],
        }
    ]

    system = [{"text": "You are a helpful assistant"}]

    modelId = "amazon.nova-lite-v1:0"

    inf_params = {"maxTokens": 300, "topP": 0.1, "temperature": 0.3}

    response = brt.converse_stream(
        modelId=modelId,
        messages=messages,
        guardrailConfig=guardrail,
        system=system,
        inferenceConfig=inf_params,
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
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "amazon"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on system prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == system[
        0
    ].get("text")

    # Assert on prompt
    assert bedrock_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.1.content"
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

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 300
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.3
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.1

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

@pytest.mark.vcr
def test_nova_cross_region_invoke(test_context, brt):

    message_list = [{"role": "user", "content": [{"text": "Tell me a joke about OpenTelemetry"}]}]
    inf_params = {"maxTokens": 500, "topP": 0.9, "topK": 20, "temperature": 0.7}
    request_body = {
        "messages": message_list,
        "inferenceConfig": inf_params,
    }

    modelId = "us.amazon.nova-lite-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = brt.invoke_model(
        body=json.dumps(request_body),
        modelId=modelId,
        accept=accept,
        contentType=contentType,
    )

    response_body = json.loads(response.get("body").read())

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name and vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "nova-lite-v1:0"
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "amazon"

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "amazon"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on prompt
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

    assert bedrock_span.attributes[
               f"{SpanAttributes.LLM_PROMPTS}.0.content"
           ] == json.dumps(message_list[0].get("content"), default=str)

    # Assert on response
    generated_text = response_body["output"]["message"]["content"]
    for i in range(0, len(generated_text)):
        assert (
                bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content"]
                == generated_text[i]["text"]
        )

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 500
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.7
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.9
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None
