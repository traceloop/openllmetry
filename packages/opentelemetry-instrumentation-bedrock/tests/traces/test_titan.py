import json

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_titan_completion(instrument_legacy, brt, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on prompt
    expected_prompt = (
        "Translate to spanish: 'Amazon Bedrock is the easiest way to build and"
        "scale generative AI applications with base models (FMs)'."
    )
    assert (
        bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
        == expected_prompt
    )

    # Assert on response
    generated_text = response_body["results"][0]["outputText"]
    assert (
        bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == generated_text
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.5
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_titan_completion_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.5
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {
            "content": "Translate to spanish: 'Amazon Bedrock is the easiest way to build and"
            + "scale generative AI applications with base models (FMs)'."
        },
    )

    # Validate the ai response
    generated_text = response_body["results"][0]["outputText"]
    choice_event = {
        "index": 0,
        "finish_reason": "FINISH",
        "message": {"content": generated_text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_titan_completion_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.5
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "FINISH",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_titan_invoke_stream(instrument_legacy, brt, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on prompt
    expected_prompt = (
        "Translate to spanish: 'Amazon Bedrock is the easiest way to build and"
        "scale generative AI applications with base models (FMs)'."
    )
    assert (
        bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
        == expected_prompt
    )

    # Assert on response
    completion_text = "".join(generated_text)
    assert (
        bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == completion_text
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.5
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_titan_invoke_stream_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.5
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    expected_prompt = (
        "Translate to spanish: 'Amazon Bedrock is the easiest way to build and"
        "scale generative AI applications with base models (FMs)'."
    )
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": expected_prompt}
    )

    # Validate the ai response
    completion_text = "".join(generated_text)
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": completion_text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_titan_invoke_stream_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.5
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_titan_converse(instrument_legacy, brt, span_exporter, log_exporter):
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
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    assert bedrock_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    ] == json.dumps(messages[0].get("content"), default=str)

    # Assert on response
    generated_text = response["output"]["message"]["content"]
    for i in range(0, len(generated_text)):
        assert (
            bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{i}.content"]
            == generated_text[i]["text"]
        )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_titan_converse_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": messages[0]["content"]}
    )

    # Validate the ai response
    generated_text = response["output"]["message"]["content"]
    choice_event = {
        "index": 0,
        "finish_reason": "guardrail_intervened",
        "message": {"content": generated_text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_titan_converse_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    brt.converse(
        modelId=modelId,
        messages=messages,
        guardrailConfig=guardrail,
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "guardrail_intervened",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_titan_converse_stream(instrument_legacy, brt, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    assert bedrock_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    ] == json.dumps(messages[0].get("content"), default=str)

    # Assert on response
    assert (
        bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == content
    )
    assert (
        bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"]
        == response_role
    )

    # Assert on usage data
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == inputTokens
    )
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == outputTokens
    )
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == inputTokens + outputTokens
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_titan_converse_stream_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    content = ""
    inputTokens = 0
    outputTokens = 0

    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                content += event["contentBlockDelta"]["delta"]["text"]

            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    inputTokens = metadata["usage"]["inputTokens"]
                    outputTokens = metadata["usage"]["outputTokens"]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on usage data
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == inputTokens
    )
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == outputTokens
    )
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == inputTokens + outputTokens
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": messages[0]["content"]}
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "guardrail_intervened",
        "message": {"content": content},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_titan_converse_stream_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    content = ""
    inputTokens = 0
    outputTokens = 0

    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                content += event["contentBlockDelta"]["delta"]["text"]

            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    inputTokens = metadata["usage"]["inputTokens"]
                    outputTokens = metadata["usage"]["outputTokens"]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on usage data
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == inputTokens
    )
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == outputTokens
    )
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == inputTokens + outputTokens
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "guardrail_intervened",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.AWS_BEDROCK.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
