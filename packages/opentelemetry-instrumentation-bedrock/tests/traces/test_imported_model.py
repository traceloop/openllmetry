import json

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_imported_model_completion(instrument_legacy, brt, span_exporter, log_exporter):
    prompt = "Explain quantum mechanics."
    payload = {"prompt": prompt, "max_tokens": 100, "topP": 2, "temperature": 0.5}
    payload_str = json.dumps(payload)
    model_arn = (
        "arn:aws:sagemaker:us-east-1:767398002385:endpoint/endpoint-quick-start-idr7y"
    )
    response = brt.invoke_model(modelId=model_arn, body=payload_str)
    data = json.loads(response["body"].read().decode("utf-8"))
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"
    imported_model_span = spans[0]

    assert (
        imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "arn:aws:sagemaker:us-east-1:767398002385:endpoint/endpoint-quick-start-idr7y"
    )
    assert (
        imported_model_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    )
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"
    assert imported_model_span.attributes.get("gen_ai.response.id") is None
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 2

    assert (
        imported_model_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == prompt
    )
    assert data is not None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_imported_model_completion_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
    prompt = "Explain quantum mechanics."
    payload = {"prompt": prompt, "max_tokens": 100, "topP": 2, "temperature": 0.5}
    payload_str = json.dumps(payload)
    model_arn = (
        "arn:aws:sagemaker:us-east-1:767398002385:endpoint/endpoint-quick-start-idr7y"
    )
    response = brt.invoke_model(modelId=model_arn, body=payload_str)
    data = json.loads(response["body"].read().decode("utf-8"))
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"
    imported_model_span = spans[0]

    assert (
        imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "arn:aws:sagemaker:us-east-1:767398002385:endpoint/endpoint-quick-start-idr7y"
    )
    assert (
        imported_model_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    )
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"
    assert imported_model_span.attributes.get("gen_ai.response.id") is None
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 2

    assert data is not None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {"content": prompt})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {"content": data["choices"][0]["text"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_imported_model_completion_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
    prompt = "Explain quantum mechanics."
    payload = {"prompt": prompt, "max_tokens": 100, "topP": 2, "temperature": 0.5}
    payload_str = json.dumps(payload)
    model_arn = (
        "arn:aws:sagemaker:us-east-1:767398002385:endpoint/endpoint-quick-start-idr7y"
    )
    response = brt.invoke_model(modelId=model_arn, body=payload_str)
    data = json.loads(response["body"].read().decode("utf-8"))
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.completion"
    imported_model_span = spans[0]

    assert (
        imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "arn:aws:sagemaker:us-east-1:767398002385:endpoint/endpoint-quick-start-idr7y"
    )
    assert (
        imported_model_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    )
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"
    assert imported_model_span.attributes.get("gen_ai.response.id") is None
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert imported_model_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 2

    assert data is not None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "length",
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
