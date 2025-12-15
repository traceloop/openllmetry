import json

import pytest
from opentelemetry.instrumentation.bedrock.prompt_caching import CacheSpanAttrs
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_anthropic_2_completion(instrument_legacy, brt, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
        == "Human: Tell me a joke about opentelemetry Assistant:"
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == completion
    )

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    # Bedrock does not return the response id for claude-v2:1
    assert anthropic_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_2_completion_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    # Bedrock does not return the response id for claude-v2:1
    assert anthropic_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Human: Tell me a joke about opentelemetry Assistant:"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop_sequence",
        "message": {"content": completion},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_2_completion_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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
    response_body.get("completion")

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    # Bedrock does not return the response id for claude-v2:1
    assert anthropic_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop_sequence",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_3_completion_complex_content(
    instrument_legacy, brt, span_exporter, log_exporter
):
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

    response = brt.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    completion = response_body.get("content")

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]
    assert json.loads(
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    ) == [
        {"type": "text", "text": "Tell me a joke about opentelemetry"},
    ]

    assert (
        json.loads(
            anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        )
        == completion
    )

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_bdrk_01Q6Z4xmUkMigo9K4qd1fshW"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_3_completion_complex_content_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    response = brt.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    completion = response_body.get("content")

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_bdrk_01Q6Z4xmUkMigo9K4qd1fshW"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {
            "content": [
                {"type": "text", "text": "Tell me a joke about opentelemetry"},
            ],
        },
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": completion},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_3_completion_complex_content_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    response = brt.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    response_body.get("content")

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_bdrk_01Q6Z4xmUkMigo9K4qd1fshW"
    )

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


@pytest.mark.vcr
def test_anthropic_3_completion_streaming(
    instrument_legacy, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]
    assert json.loads(
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    ) == [
        {"type": "text", "text": "Tell me a joke about opentelemetry"},
    ]

    assert json.loads(
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    ) == [
        {
            "type": "text",
            "text": completion,
        }
    ]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_bdrk_014eJfxWXNnxFKhmuiT8FYf7"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_3_completion_streaming_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_bdrk_014eJfxWXNnxFKhmuiT8FYf7"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": [{"text": "Tell me a joke about opentelemetry", "type": "text"}]},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.get("body")._accumulating_body.get("content")},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_3_completion_streaming_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_bdrk_014eJfxWXNnxFKhmuiT8FYf7"
    )

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
def test_anthropic_3_completion_string_content(
    instrument_legacy, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]
    assert (
        json.loads(anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"])
        == "Tell me a joke about opentelemetry"
    )

    assert (
        json.loads(
            anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        )
        == completion
    )

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_bdrk_01WR9VHqpyBzBhzgwCDapaQD"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_3_completion_string_content_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_bdrk_01WR9VHqpyBzBhzgwCDapaQD"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": completion},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_3_completion_string_content_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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

    json.loads(response.get("body").read())

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_bdrk_01WR9VHqpyBzBhzgwCDapaQD"
    )

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


@pytest.mark.vcr
def test_anthropic_cross_region(instrument_legacy, brt, span_exporter, log_exporter):
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Human: Tell me a joke about opentelemetry Assistant:"},
            ],
        },
    ]
    response = brt.converse(
        modelId="arn:aws:bedrock:us-east-1:012345678901:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=messages,
        inferenceConfig=inference_config,
    )
    completion = response["output"]["message"]["content"][0]["text"]

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1

    anthropic_span = spans[0]
    assert anthropic_span.name == "bedrock.converse"

    # Assert on model name and vendor
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "claude-3-7-sonnet-20250219-v1"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    assert anthropic_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    ] == json.dumps(messages[0]["content"])
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == completion
    )

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 20
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    # Bedrock does not return the response id for claude-v2:1
    assert anthropic_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_cross_region_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Human: Tell me a joke about opentelemetry Assistant:"},
            ],
        },
    ]
    response = brt.converse(
        modelId="arn:aws:bedrock:us-east-1:012345678901:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=messages,
        inferenceConfig=inference_config,
    )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1

    anthropic_span = spans[0]
    assert anthropic_span.name == "bedrock.converse"

    # Assert on model name and vendor
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "claude-3-7-sonnet-20250219-v1"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 20
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    # Bedrock does not return the response id for claude-v2:1
    assert anthropic_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {
            "content": [
                {"text": "Human: Tell me a joke about opentelemetry Assistant:"},
            ]
        },
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": response["output"]["message"]["content"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_cross_region_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Human: Tell me a joke about opentelemetry Assistant:"},
            ],
        },
    ]
    brt.converse(
        modelId="arn:aws:bedrock:us-east-1:012345678901:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=messages,
        inferenceConfig=inference_config,
    )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1

    anthropic_span = spans[0]
    assert anthropic_span.name == "bedrock.converse"

    # Assert on model name and vendor
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "claude-3-7-sonnet-20250219-v1"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"

    assert anthropic_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 20
    assert anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + anthropic_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ) == anthropic_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    # Bedrock does not return the response id for claude-v2:1
    assert anthropic_span.attributes.get("gen_ai.response.id") is None

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


@pytest.mark.vcr
def test_prompt_cache(instrument_legacy, brt, span_exporter, log_exporter):
    def prompt_caching_call(brt):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": "very very long system prompt",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How do I write js?",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1,
            "top_p": 0.1,
            "stop_sequences": ["stop"],
            "top_k": 250,
        }
        return brt.invoke_model(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            body=json.dumps(body),
        )

    prompt_caching_call(brt)
    prompt_caching_call(brt)

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 2

    # first writes, second reads
    assert spans[0].attributes.get(CacheSpanAttrs.CACHED) == "write"
    assert spans[1].attributes.get(CacheSpanAttrs.CACHED) == "read"


@pytest.mark.vcr
def test_anthropic_converse_stream_with_tool_use(
    instrument_legacy, brt, span_exporter, log_exporter
):
    """Test converse_stream with tool use to validate handling of non-text contentBlockDelta events."""
    messages = [
        {
            "role": "user",
            "content": [{"text": "What is the weather in Barcelona?"}],
        }
    ]

    tools = [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name",
                            }
                        },
                        "required": ["location"],
                    }
                },
            }
        }
    ]

    tool_config = {"tools": tools}
    inf_params = {"maxTokens": 300, "temperature": 0.5}

    response = brt.converse_stream(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        toolConfig=tool_config,
        inferenceConfig=inf_params,
    )

    stream = response.get("stream")
    content = ""
    tool_use_found = False

    for event in stream:
        # Test should handle contentBlockDelta events without text field
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            # Only append if text exists (validates the fix)
            if "text" in delta:
                content += delta["text"]
            # Check if we got a toolUse delta
            if "toolUse" in delta:
                tool_use_found = True

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    bedrock_span = spans[0]

    # Assert on model name
    assert (
        bedrock_span.attributes.get("gen_ai.request.model")
        == "claude-3-sonnet-20240229-v1:0"
    )

    # Assert on vendor
    assert bedrock_span.attributes.get("gen_ai.system") == "AWS"

    # Assert on request type
    assert bedrock_span.attributes.get("llm.request.type") == "chat"

    # tool use should have been triggered
    # (This test validates that non-text deltas don't crash the instrumentation)
    assert tool_use_found is True

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


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
