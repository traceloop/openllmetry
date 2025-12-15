import json

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


JOKE_SCHEMA = {
    "type": "object",
    "properties": {
        "joke": {
            "type": "string",
            "description": "A joke about OpenTelemetry"
        },
        "rating": {
            "type": "integer",
            "description": "Rating of the joke from 1 to 10"
        }
    },
    "required": ["joke", "rating"],
    "additionalProperties": False
}

OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": JOKE_SCHEMA
}


@pytest.mark.vcr
def test_anthropic_structured_outputs_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter
):
    response = anthropic_client.beta.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        betas=["structured-outputs-2025-11-13"],
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry and rate it from 1 to 10"
            }
        ],
        output_format=OUTPUT_FORMAT
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "anthropic.chat"

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry and rate it from 1 to 10"
    )
    assert anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )

    assert SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA in anthropic_span.attributes
    schema_attr = json.loads(
        anthropic_span.attributes[SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA]
    )
    assert "properties" in schema_attr
    assert "joke" in schema_attr["properties"]
    assert "rating" in schema_attr["properties"]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "claude-sonnet-4-5-20250929"
    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL) == "claude-sonnet-4-5-20250929"

    response_json = json.loads(response.content[0].text)
    assert "joke" in response_json
    assert "rating" in response_json

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_anthropic_structured_outputs_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter
):
    response = anthropic_client.beta.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        betas=["structured-outputs-2025-11-13"],
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry and rate it from 1 to 10"
            }
        ],
        output_format=OUTPUT_FORMAT
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "anthropic.chat"

    response_json = json.loads(response.content[0].text)
    assert "joke" in response_json
    assert "rating" in response_json

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2


@pytest.mark.vcr
def test_anthropic_structured_outputs_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter
):
    response = anthropic_client.beta.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        betas=["structured-outputs-2025-11-13"],
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry and rate it from 1 to 10"
            }
        ],
        output_format=OUTPUT_FORMAT
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "anthropic.chat"

    response_json = json.loads(response.content[0].text)
    assert "joke" in response_json
    assert "rating" in response_json

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2
