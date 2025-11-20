import json

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_ai21_j2_completion_string_content(
    instrument_legacy, brt, span_exporter, log_exporter
):
    prompt = (
        "Translate to spanish: 'Amazon Bedrock is the easiest way to build and"
        + "scale generative AI applications with base models (FMs)'."
    )

    body = json.dumps(
        {
            "prompt": prompt,
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == len(
        response_body.get("prompt").get("tokens")
    )
    assert meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == len(
        response_body.get("completions")[0].get("data").get("tokens")
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        + meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
    )
    # It is apparently always 1234, but for the sake of consistency,
    # we should not assert on it.
    assert meta_span.attributes.get("gen_ai.response.id") == 1234

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_ai21_j2_completion_string_content_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
    prompt = (
        "Translate to spanish: 'Amazon Bedrock is the easiest way to build and"
        + "scale generative AI applications with base models (FMs)'."
    )

    body = json.dumps(
        {
            "prompt": prompt,
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == len(
        response_body.get("prompt").get("tokens")
    )
    assert meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == len(
        response_body.get("completions")[0].get("data").get("tokens")
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        + meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
    )
    # It is apparently always 1234, but for the sake of consistency,
    # we should not assert on it.
    assert meta_span.attributes.get("gen_ai.response.id") == 1234

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {"content": prompt})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "endoftext",
        "message": {"content": response_body["completions"][0]["data"]["text"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_ai21_j2_completion_string_content_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == len(
        response_body.get("prompt").get("tokens")
    )
    assert meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == len(
        response_body.get("completions")[0].get("data").get("tokens")
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        + meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
    )
    # It is apparently always 1234, but for the sake of consistency,
    # we should not assert on it.
    assert meta_span.attributes.get("gen_ai.response.id") == 1234

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "endoftext",
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
