import json

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr()
def test_sagemaker_completion_string_content_legacy(
    instrument_legacy, span_exporter, smrt, log_exporter
):
    endpoint_name = "my-llama2-endpoint"
    prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden  What should I do? [/INST]"""
    # Create request body.
    body = json.dumps(
        {
            "inputs": prompt,
            "parameters": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 128},
        }
    )

    smrt.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=body,
        ContentType="application/json",
    )

    spans = span_exporter.get_finished_spans()

    meta_span = spans[0]
    assert meta_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == endpoint_name
    assert meta_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == body

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr()
def test_sagemaker_completion_string_content_with_events_with_content(
    instrument_with_content, span_exporter, smrt, log_exporter
):
    endpoint_name = "my-llama2-endpoint"
    prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden  What should I do? [/INST]"""
    # Create request body.
    body = json.dumps(
        {
            "inputs": prompt,
            "parameters": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 128},
        }
    )

    response = smrt.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=body,
        ContentType="application/json",
    )

    spans = span_exporter.get_finished_spans()

    meta_span = spans[0]
    assert meta_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == endpoint_name

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {"content": prompt})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {
            "content": json.loads(response["Body"].read())[0].get("generated_text")
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr()
def test_sagemaker_completion_string_content_with_events_with_no_content(
    instrument_with_no_content, span_exporter, smrt, log_exporter
):
    endpoint_name = "my-llama2-endpoint"
    prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden  What should I do? [/INST]"""
    # Create request body.
    body = json.dumps(
        {
            "inputs": prompt,
            "parameters": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 128},
        }
    )

    smrt.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=body,
        ContentType="application/json",
    )

    spans = span_exporter.get_finished_spans()

    meta_span = spans[0]
    assert meta_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == endpoint_name

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


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "sagemaker"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
