import json

import pytest
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_anthropic_2_completion(test_context, brt):
    body = json.dumps(
        {
            "prompt": "Human: Tell me a joke about opentelemetry Assistant:",
            "max_tokens_to_sample": 200,
            "temperature": 0.5,
        }
    )

    response = brt.invoke_model(
        body=body,
        modelId="anthropic.claude-v2:1",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    completion = response_body.get("completion")

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == "Human: Tell me a joke about opentelemetry Assistant:"
    )
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == completion
    )

    assert anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 13
    assert anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    ) == anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    )


@pytest.mark.vcr
def test_anthropic_3_completion_complex_content(test_context, brt):
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tell me a joke about opentelemetry"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Why did the microservice bring OpenTelemetry to the party?"}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Because it knew how to keep track of all the span-taneous events!"}
                    ],
                },
            ],
            "system": "You're a comedian",
            "max_tokens": 200,
            "temperature": 0.5,
            "anthropic_version": "bedrock-2023-05-31",
        }
    )

    response = brt.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    completion = response_body.get("content")

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "You're a comedian"

    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == "Tell me a joke about opentelemetry"

    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "assistant"
    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"] == "Why did the microservice bring OpenTelemetry to the party?"

    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.3.role"] == "user"
    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.3.content"] == "Because it knew how to keep track of all the span-taneous events!"

    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    ) == completion[0].get("text")

    assert anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    ) == anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    )


@pytest.mark.vcr
def test_anthropic_3_completion_streaming(test_context, brt):
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tell me a joke about opentelemetry"},
                    ],
                }
            ],
            "max_tokens": 200,
            "temperature": 0.5,
            "anthropic_version": "bedrock-2023-05-31",
        }
    )

    response = brt.invoke_model_with_response_stream(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    completion = ""
    for event in response.get("body"):
        chunk = event.get("chunk")
        if chunk:
            decoded_chunk = json.loads(chunk.get("bytes").decode())
            if "delta" in decoded_chunk:
                completion += decoded_chunk.get("delta").get("text") or ""

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]
    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Tell me a joke about opentelemetry"

    assert anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == completion

    assert anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    ) == anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    )


@pytest.mark.vcr
def test_anthropic_3_completion_string_content(test_context, brt):
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a joke about opentelemetry",
                }
            ],
            "max_tokens": 200,
            "temperature": 0.5,
            "anthropic_version": "bedrock-2023-05-31",
        }
    )

    response = brt.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    completion = response_body.get("content")

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    for key, value in anthropic_span.attributes.items():
        print(key, value)

    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )

    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == completion[0].get("text")
    )

    assert anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    ) == anthropic_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    )
