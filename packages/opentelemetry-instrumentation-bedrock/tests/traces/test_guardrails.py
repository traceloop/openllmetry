import json

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.instrumentation.bedrock.span_utils import PROMPT_FILTER_KEY, CONTENT_FILTER_KEY
from opentelemetry.semconv._incubating.attributes.aws_attributes import (
    AWS_BEDROCK_GUARDRAIL_ID
)

guardrailId = "5zwrmdlsra2e"
guardrailVersion = "DRAFT"


@pytest.mark.vcr
def test_guardrail_invoke(instrument_legacy, brt, span_exporter, log_exporter):
    body = json.dumps(
        {
            "inputText": "Tell me a joke about opentelemetry",
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.5,
            },
        }
    )

    modelId = "amazon.titan-text-express-v1"
    accept = "application/json"
    contentType = "application/json"

    response = brt.invoke_model(
        body=body,
        modelId=modelId,
        accept=accept,
        contentType=contentType,
        guardrailIdentifier=guardrailId,
        guardrailVersion=guardrailVersion
    )

    _ = json.loads(response.get("body").read())

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

    # Assert on guardrail data
    assert bedrock_span.attributes[AWS_BEDROCK_GUARDRAIL_ID] == f"{guardrailId}:{guardrailVersion}"
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_FILTER_KEY}"] != ""
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{CONTENT_FILTER_KEY}"] != ""

    input_guardrail = json.loads(bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_FILTER_KEY}"])
    output_guardrail = json.loads(bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{CONTENT_FILTER_KEY}"])

    assert input_guardrail["topic"] == []
    assert input_guardrail["content"] == []
    assert input_guardrail["words"] == []
    assert input_guardrail["sensitive"]["pii"] == []
    assert input_guardrail["sensitive"]["regex"] == []

    assert len(output_guardrail) == 1
    output_guardrail = output_guardrail[0]
    assert output_guardrail["topic"] == []
    assert output_guardrail["content"] == []
    assert output_guardrail["words"] == []
    assert output_guardrail["sensitive"]["pii"] == []
    assert output_guardrail["sensitive"]["regex"] == ["Account Number"]


@pytest.mark.vcr
def test_guardrail_invoke_stream(instrument_legacy, brt, span_exporter, log_exporter):
    body = json.dumps(
        {
            "inputText": "How do I play tennis in Japan?",
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.5,
            },
        }
    )

    modelId = "amazon.titan-text-express-v1"
    accept = "application/json"
    contentType = "application/json"

    response = brt.invoke_model_with_response_stream(
        body=body,
        modelId=modelId,
        accept=accept,
        contentType=contentType,
        guardrailIdentifier=guardrailId,
        guardrailVersion=guardrailVersion
    )
    # consume events
    stream = response.get('body')
    if stream:
        for event in stream:
            continue

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

    # Assert on guardrail data
    assert bedrock_span.attributes[AWS_BEDROCK_GUARDRAIL_ID] == f"{guardrailId}:{guardrailVersion}"
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_FILTER_KEY}"] != ""
    assert bedrock_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.{CONTENT_FILTER_KEY}") is None

    input_guardrail = json.loads(bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_FILTER_KEY}"])

    assert input_guardrail["topic"] == ["topic-1"]
    assert input_guardrail["content"] == []
    assert input_guardrail["words"] == []
    assert input_guardrail["sensitive"]["pii"] == []
    assert input_guardrail["sensitive"]["regex"] == []


@pytest.mark.vcr
def test_guardrail_converse(
    instrument_with_content, brt, span_exporter, log_exporter
):
    guardrail = {
        'guardrailIdentifier': guardrailId,
        'guardrailVersion': guardrailVersion,
        'trace': 'enabled'
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": "what is the capital of Italy?"
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "text": "what is the capital of Japan?"
                },
            ],
        },
    ]

    modelId = "amazon.titan-text-express-v1"

    _ = brt.converse(
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

    # Assert on guardrail data
    assert bedrock_span.attributes[AWS_BEDROCK_GUARDRAIL_ID] == f"{guardrailId}:{guardrailVersion}"
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_FILTER_KEY}"] != ""
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{CONTENT_FILTER_KEY}"] != ""

    input_guardrail = json.loads(bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_FILTER_KEY}"])
    output_guardrail = json.loads(bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{CONTENT_FILTER_KEY}"])

    assert input_guardrail["topic"] == []
    assert input_guardrail["content"] == []
    assert input_guardrail["words"] == []
    assert input_guardrail["sensitive"]["pii"] == []
    assert input_guardrail["sensitive"]["regex"] == []

    assert len(output_guardrail) == 1
    output_guardrail = output_guardrail[0]
    assert output_guardrail["topic"] == []
    assert output_guardrail["content"] == []
    assert output_guardrail["words"] == []
    assert output_guardrail["sensitive"]["pii"] == ["ADDRESS"]
    assert output_guardrail["sensitive"]["regex"] == []


@pytest.mark.vcr
def test_guardrail_converse_stream(
        instrument_with_content, brt, span_exporter, log_exporter
):
    guardrail = {
        'guardrailIdentifier': guardrailId,
        'guardrailVersion': guardrailVersion,
        'trace': 'enabled'
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": "what is the capital of Italy?"
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "text": "what is the capital of Japan?"
                },
            ],
        },
    ]

    modelId = "amazon.titan-text-express-v1"

    response = brt.converse_stream(
        modelId=modelId,
        messages=messages,
        guardrailConfig=guardrail,
    )
    # consume events
    stream = response.get('stream')
    if stream:
        for event in stream:
            continue

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

    # Assert on guardrail data
    assert bedrock_span.attributes[AWS_BEDROCK_GUARDRAIL_ID] == f"{guardrailId}:{guardrailVersion}"
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_FILTER_KEY}"] != ""
    assert bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{CONTENT_FILTER_KEY}"] != ""

    input_guardrail = json.loads(bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.{PROMPT_FILTER_KEY}"])
    output_guardrail = json.loads(bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{CONTENT_FILTER_KEY}"])

    assert input_guardrail["topic"] == []
    assert input_guardrail["content"] == []
    assert input_guardrail["words"] == []
    assert input_guardrail["sensitive"]["pii"] == []
    assert input_guardrail["sensitive"]["regex"] == []

    assert len(output_guardrail) == 1
    output_guardrail = output_guardrail[0]
    assert output_guardrail["topic"] == []
    assert output_guardrail["content"] == []
    assert output_guardrail["words"] == []
    assert output_guardrail["sensitive"]["pii"] == ["ADDRESS"]
    assert output_guardrail["sensitive"]["regex"] == []
