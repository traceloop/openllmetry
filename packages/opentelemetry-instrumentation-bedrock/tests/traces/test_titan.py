import json

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.instrumentation.bedrock import PromptCaching
from opentelemetry.instrumentation.bedrock.prompt_caching import CacheSpanAttrs


def get_metric(resource_metrics, name):
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    return metric
    raise Exception(f"No metric found with name {name}")


def assert_metric(reader, usage, is_read=False):
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    m = get_metric(resource_metrics, PromptCaching.LLM_BEDROCK_PROMPT_CACHING)
    # This check is now more specific to handle cumulative metrics
    found_read = False
    found_write = False
    for data_point in m.data.data_points:
        if data_point.attributes[CacheSpanAttrs.TYPE] == "read":
            found_read = True
            assert data_point.value == usage["cache_read_input_tokens"]
        elif data_point.attributes[CacheSpanAttrs.TYPE] == "write":
            found_write = True
            assert data_point.value == usage["cache_creation_input_tokens"]
    
    if is_read:
        assert found_read
    else:
        assert found_write
        

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
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.5
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
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.5
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
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.5
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
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.5
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
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

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
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

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
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

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
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "titan-text-express-v1"
    )

    # Assert on vendor
    assert bedrock_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

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
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.AWS_BEDROCK.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


@pytest.mark.vcr
def test_titan_converse_with_caching(instrument_legacy, brt, span_exporter, reader):

    # --- 1. First call (will write to cache) ---
    response_write = brt.converse(
        modelId="amazon.titan-text-express-v1",
        messages=[{"role": "user", "content": [{"text": "Hello, this is a test prompt for caching."}]}],
        inferenceConfig={"maxTokens": 50},
        additionalModelRequestFields={"cacheControl": {"type": "ephemeral"}},
    )
    usage_write = response_write["usage"]
    assert usage_write["cache_read_input_tokens"] == 0
    assert usage_write["cache_creation_input_tokens"] > 0

    # --- 2. Second call (will read from cache) ---
    response_read = brt.converse(
        modelId="amazon.titan-text-express-v1",
        messages=[{"role": "user", "content": [{"text": "Hello, this is a test prompt for caching."}]}],
        inferenceConfig={"maxTokens": 50},
        additionalModelRequestFields={"cacheControl": {"type": "ephemeral"}},
    )
    usage_read = response_read["usage"]
    assert usage_read["cache_read_input_tokens"] > 0
    assert usage_read["cache_creation_input_tokens"] == 0

    # --- 3. Assertions ---
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    # Assertions for the first span (cache write)
    span_write = spans[0]
    assert span_write.name == "bedrock.converse"
    attributes_write = span_write.attributes
    assert attributes_write[SpanAttributes.LLM_REQUEST_MODEL] == "titan-text-express-v1"
    assert attributes_write[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] > 0
    assert attributes_write[CacheSpanAttrs.CACHED] == "write"
    assert attributes_write[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == usage_write["cache_creation_input_tokens"]

    # Assertions for the second span (cache read)
    span_read = spans[1]
    assert span_read.name == "bedrock.converse"
    attributes_read = span_read.attributes
    assert attributes_read[SpanAttributes.LLM_REQUEST_MODEL] == "titan-text-express-v1"
    assert attributes_read[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] > 0
    assert attributes_read[CacheSpanAttrs.CACHED] == "read"
    assert attributes_read[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == usage_read["cache_read_input_tokens"]

    # Assert metrics (we need to combine usage for cumulative assertion)
    cumulative_usage = {
        "cache_creation_input_tokens": usage_write["cache_creation_input_tokens"],
        "cache_read_input_tokens": usage_read["cache_read_input_tokens"],
    }
    assert_metric(reader, cumulative_usage, is_read=False)
    assert_metric(reader, cumulative_usage, is_read=True)