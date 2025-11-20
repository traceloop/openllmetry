import json

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_cohere_completion(instrument_legacy, brt, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "command-text-v14"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on prompt
    assert bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"] == prompt

    # Assert on response
    generated_text = response_body["generations"][0]["text"]
    assert (
        bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == generated_text
    )
    assert (
        bedrock_span.attributes.get("gen_ai.response.id")
        == "3266ca30-473c-4491-b6ef-5b1f033798d2"
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.5

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_completion_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "command-text-v14"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on response
    assert (
        bedrock_span.attributes.get("gen_ai.response.id")
        == "3266ca30-473c-4491-b6ef-5b1f033798d2"
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.5

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "COMPLETE",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_cohere_completion_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "command-text-v14"
    )

    # Assert on vendor
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    # Assert on request type
    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"

    # Assert on response
    generated_text = response_body["generations"][0]["text"]
    assert (
        bedrock_span.attributes.get("gen_ai.response.id")
        == "3266ca30-473c-4491-b6ef-5b1f033798d2"
    )

    # Assert on other request parameters
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.5

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {"content": prompt})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "COMPLETE",
        "message": {"content": generated_text},
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
