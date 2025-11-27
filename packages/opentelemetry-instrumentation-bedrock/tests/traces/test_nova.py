import json

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_nova_completion(instrument_legacy, brt, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on system prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert bedrock_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    ] == system_list[0].get("text")

    # Assert on prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "user"

    assert bedrock_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"
    ] == json.dumps(message_list[0].get("content"), default=str)

    # Assert on response
    generated_text = response_body["output"]["message"]["content"]
    for i in range(0, len(generated_text)):
        assert (
            bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{i}.content"]
            == generated_text[i]["text"]
        )

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 500
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_nova_completion_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on response
    generated_text = response_body["output"]["message"]["content"]

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 500
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.system.message",
        {"content": "tell me a very two sentence story."},
    )

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": [{"text": "A camping trip"}]},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": generated_text},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_nova_completion_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    brt.invoke_model(
        body=json.dumps(request_body),
        modelId=modelId,
        accept=accept,
        contentType=contentType,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 500
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.system.message", {})

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_nova_invoke_stream(instrument_legacy, brt, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on system prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert bedrock_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    ] == system_list[0].get("text")

    # Assert on prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "user"

    assert bedrock_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"
    ] == json.dumps(message_list[0].get("content"), default=str)

    # Assert on response
    completion_msg = "".join(generated_text)
    assert (
        bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == completion_msg
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS
    ] == inf_params.get("maxTokens")
    assert bedrock_span.attributes[
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE
    ] == inf_params.get("temperature")
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == inf_params.get(
        "topP"
    )
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_nova_invoke_stream_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on response
    completion_msg = "".join(generated_text)

    # Assert on other request parameters
    assert bedrock_span.attributes[
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS
    ] == inf_params.get("maxTokens")
    assert bedrock_span.attributes[
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE
    ] == inf_params.get("temperature")
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == inf_params.get(
        "topP"
    )
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.system.message",
        {"content": "tell me a very two sentence story."},
    )

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": [{"text": "A camping trip"}]},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": completion_msg},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_nova_invoke_stream_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS
    ] == inf_params.get("maxTokens")
    assert bedrock_span.attributes[
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE
    ] == inf_params.get("temperature")
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == inf_params.get(
        "topP"
    )
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.system.message", {})

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_nova_converse(instrument_legacy, brt, span_exporter, log_exporter):
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
                            "text": "Tokyo is the capital of Japan."
                            + "The Greater Tokyo area is the most populous metropolitan area in the world.",
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on system prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == system[
        0
    ].get("text")

    # Assert on prompt
    assert bedrock_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"
    ] == json.dumps(messages[0].get("content"), default=str)

    # Assert on response
    generated_text = response["output"]["message"]["content"]
    for i in range(0, len(generated_text)):
        assert (
            bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{i}.content"]
            == generated_text[i]["text"]
        )

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 300
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.1

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_nova_converse_with_events_with_content(
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
                            "text": "Tokyo is the capital of Japan."
                            + "The Greater Tokyo area is the most populous metropolitan area in the world.",
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 300
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.1

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.system.message",
        {"content": "You are a helpful assistant"},
    )

    # Validate user message Event
    user_message_log = logs[1]
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
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_nova_converse_with_events_with_no_content(
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
                            "text": "Tokyo is the capital of Japan."
                            + "The Greater Tokyo area is the most populous metropolitan area in the world.",
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

    brt.converse(
        modelId=modelId,
        messages=messages,
        guardrailConfig=guardrail,
        system=system,
        inferenceConfig=inf_params,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 300
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.1

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.system.message", {})

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "guardrail_intervened",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_nova_converse_stream(instrument_legacy, brt, span_exporter, log_exporter):
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
                            "text": "Tokyo is the capital of Japan."
                            + "The Greater Tokyo area is the most populous metropolitan area in the world.",
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on system prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == system[
        0
    ].get("text")

    # Assert on prompt
    assert bedrock_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"
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

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 300
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.1

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
def test_nova_converse_stream_with_events_with_content(
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
                            "text": "Tokyo is the capital of Japan."
                            + "The Greater Tokyo area is the most populous metropolitan area in the world.",
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
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 300
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.1

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
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.system.message",
        {"content": "You are a helpful assistant"},
    )

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": messages[0]["content"]}
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "guardrail_intervened",
        "message": {"content": content},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_nova_converse_stream_with_events_with_no_content(
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
                            "text": "Tokyo is the capital of Japan."
                            + "The Greater Tokyo area is the most populous metropolitan area in the world.",
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
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 300
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.1

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
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.system.message", {})

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "guardrail_intervened",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_nova_cross_region_invoke(instrument_legacy, brt, span_exporter, log_exporter):
    message_list = [
        {"role": "user", "content": [{"text": "Tell me a joke about OpenTelemetry"}]}
    ]
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name and vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"

    assert bedrock_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    ] == json.dumps(message_list[0].get("content"), default=str)

    # Assert on response
    generated_text = response_body["output"]["message"]["content"]
    for i in range(0, len(generated_text)):
        assert (
            bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{i}.content"]
            == generated_text[i]["text"]
        )

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 500
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_nova_cross_region_invoke_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
    message_list = [
        {"role": "user", "content": [{"text": "Tell me a joke about OpenTelemetry"}]}
    ]
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name and vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 500
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    # There is no response id for Amazon Titan models in the response body,
    # only request id in the response.
    assert bedrock_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": message_list[0]["content"]}
    )

    # Validate the ai response
    generated_text = response_body["output"]["message"]["content"]
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": generated_text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_nova_cross_region_invoke_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
    message_list = [
        {"role": "user", "content": [{"text": "Tell me a joke about OpenTelemetry"}]}
    ]
    inf_params = {"maxTokens": 500, "topP": 0.9, "topK": 20, "temperature": 0.7}
    request_body = {
        "messages": message_list,
        "inferenceConfig": inf_params,
    }

    modelId = "us.amazon.nova-lite-v1:0"
    accept = "application/json"
    contentType = "application/json"

    brt.invoke_model(
        body=json.dumps(request_body),
        modelId=modelId,
        accept=accept,
        contentType=contentType,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name and vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 500
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
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
        "finish_reason": "end_turn",
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
