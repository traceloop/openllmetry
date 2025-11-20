import asyncio
import base64
import json
from pathlib import Path

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from .utils import verify_metrics

image_content_block = {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": base64.b64encode(
            open(
                Path(__file__).parent.joinpath("data/logo.jpg"),
                "rb",
            ).read()
        ).decode("utf-8"),
    },
}


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current time in a given time zone",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        },
    },
]


@pytest.mark.vcr
def test_anthropic_message_create_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01TPXhkPo8jy6yQMrMhjpiAE"
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(resource_metrics, "claude-3-opus-20240229")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_anthropic_message_create_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-opus-20240229")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {"content": "Tell me a joke about OpenTelemetry"}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": "Sure, here's a joke about OpenTelemetry:\n\nWhy did the developer adopt OpenTelemetry?"
            "\nBecause they wanted to trace their steps and meter their progress!\n\nExplanation:\nOpenTelemetry "
            "is an open-source observability framework that provides a set of APIs, libraries, and tools to "
            "instrument, generate, collect, and export telemetry data (traces, metrics, and logs) for distributed "
            'systems. The joke plays on the words "trace" and "meter," which are key concepts in OpenTelemetry.\n\n- '
            '"Trace their steps" refers to distributed tracing, which involves recording the path of a request as it '
            'travels through a system of microservices.\n- "Meter their progress" refers to metrics, which are '
            "quantitative measurements of a system's performance and behavior.\n\nThe joke suggests that the "
            "developer adopted OpenTelemetry to gain better visibility and understanding of their application's "
            "behavior and performance, just like how one might trace their steps and meter their progress in real "
            "life."
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_message_create_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-opus-20240229")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_multi_modal_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter
):
    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see?",
                    },
                    image_content_block,
                ],
            },
        ],
        model="claude-3-opus-20240229",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    ] == json.dumps(
        [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "/some/url"}},
        ]
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1381
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01B37ySLPzYj8KY6uZmiPoxd"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_anthropic_multi_modal_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter
):
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What do you see?",
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(
                        open(
                            Path(__file__).parent.joinpath("data/logo.jpg"),
                            "rb",
                        ).read()
                    ).decode("utf-8"),
                },
            },
        ],
    }
    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[user_message],
        model="claude-3-opus-20240229",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1381
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": response.content[0].text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_multi_modal_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter
):
    anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see?",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(
                                open(
                                    Path(__file__).parent.joinpath("data/logo.jpg"),
                                    "rb",
                                ).read()
                            ).decode("utf-8"),
                        },
                    },
                ],
            },
        ],
        model="claude-3-opus-20240229",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1381
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_image_with_history(
    instrument_legacy, anthropic_client, span_exporter, log_exporter
):
    system_message = "You are a helpful assistant. Be concise and to the point."
    user_message1 = {
        "role": "user",
        "content": "Are you capable of describing an image?",
    }
    user_message2 = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see?"},
            image_content_block,
        ],
    }

    response1 = anthropic_client.messages.create(
        max_tokens=1024,
        model="claude-3-5-haiku-latest",
        system=system_message,
        messages=[
            user_message1,
        ],
    )

    response2 = anthropic_client.messages.create(
        max_tokens=1024,
        model="claude-3-5-haiku-latest",
        system=system_message,
        messages=[
            user_message1,
            {"role": "assistant", "content": response1.content},
            user_message2,
        ],
    )

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)
    assert (
        spans[0].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == system_message
    )
    assert spans[0].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert (
        spans[0].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
        == "Are you capable of describing an image?"
    )
    assert spans[0].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "user"
    assert (
        spans[0].attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response1.content[0].text
    )
    assert (
        spans[0].attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"] == "assistant"
    )
    assert (
        spans[0].attributes.get("gen_ai.response.id") == "msg_01Ctc62hUPvikvYASXZqTo9q"
    )

    assert (
        spans[1].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == system_message
    )
    assert spans[1].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert (
        spans[1].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
        == "Are you capable of describing an image?"
    )
    assert spans[1].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "user"
    assert (
        spans[1].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.content"]
        == response1.content[0].text
    )
    assert spans[1].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.role"] == "assistant"
    assert json.loads(
        spans[1].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.3.content"]
    ) == [
        {"type": "text", "text": "What do you see?"},
        {"type": "image_url", "image_url": {"url": "/some/url"}},
    ]
    assert spans[1].attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.3.role"] == "user"

    assert (
        spans[1].attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response2.content[0].text
    )
    assert (
        spans[1].attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"] == "assistant"
    )
    assert (
        spans[1].attributes.get("gen_ai.response.id") == "msg_01EtAvxHCWn5jjdUCnG4wEAd"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_async_multi_modal_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter
):
    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see?",
                    },
                    image_content_block,
                ],
            },
        ],
        model="claude-3-opus-20240229",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    ] == json.dumps(
        [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "/some/url"}},
        ]
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1311
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01DWnmUo9hWk4Fk7V7Ddfa2w"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_async_multi_modal_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter
):
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What do you see?",
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(
                        open(
                            Path(__file__).parent.joinpath("data/logo.jpg"),
                            "rb",
                        ).read()
                    ).decode("utf-8"),
                },
            },
        ],
    }
    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[user_message],
        model="claude-3-opus-20240229",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1311
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": response.content[0].text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_async_multi_modal_with_events_with_no_content(
    instrument_with_no_content, async_anthropic_client, span_exporter, log_exporter
):
    await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see?",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(
                                open(
                                    Path(__file__).parent.joinpath("data/logo.jpg"),
                                    "rb",
                                ).read()
                            ).decode("utf-8"),
                        },
                    },
                ],
            },
        ],
        model="claude-3-opus-20240229",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1311
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
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
def test_anthropic_message_streaming_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-haiku-20240307",
        stream=True,
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response_content = ""
    for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01MXWxhWoPSgrYhjTuMDM6F1"
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(resource_metrics, "claude-3-haiku-20240307")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_anthropic_message_streaming_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    user_message = {
        "role": "user",
        "content": "Tell me a joke about OpenTelemetry",
    }

    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[user_message],
        model="claude-3-haiku-20240307",
        stream=True,
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response_content = ""
    for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(resource_metrics, "claude-3-haiku-20240307")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "text",
                "content": response_content,
            }
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_message_streaming_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-haiku-20240307",
        stream=True,
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response_content = ""
    for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(resource_metrics, "claude-3-haiku-20240307")

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
@pytest.mark.asyncio
async def test_async_anthropic_message_create_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter, reader
):
    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01UFDDjsFn5BPQnfNwmsMnAY"
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-opus-20240229")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_create_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter, reader
):
    user_message = {
        "role": "user",
        "content": "Tell me a joke about OpenTelemetry",
    }
    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[user_message],
        model="claude-3-opus-20240229",
    )
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-opus-20240229")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": response.content[0].text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_create_with_events_with_no_content(
    instrument_with_no_content,
    async_anthropic_client,
    span_exporter,
    log_exporter,
    reader,
):
    await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-opus-20240229")

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
@pytest.mark.asyncio
async def test_async_anthropic_message_streaming_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter, reader
):
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-haiku-20240307",
        stream=True,
    )
    response_content = ""
    async for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_016o6A7zDmgjucf5mWv1rrPD"
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-haiku-20240307")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_streaming_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter, reader
):
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    user_message = {
        "role": "user",
        "content": "Tell me a joke about OpenTelemetry",
    }

    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[user_message],
        model="claude-3-haiku-20240307",
        stream=True,
    )
    response_content = ""
    async for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-haiku-20240307")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": {"type": "text", "content": response_content}},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_streaming_with_events_with_no_content(
    instrument_with_no_content,
    async_anthropic_client,
    span_exporter,
    log_exporter,
    reader,
):
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-haiku-20240307",
        stream=True,
    )
    response_content = ""
    async for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-haiku-20240307")

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
def test_anthropic_tools_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {
                "role": "user",
                "content": "What is the weather like right now in New York? Also what time is it there now?",
            }
        ],
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 514
    assert (
        anthropic_span.attributes["gen_ai.usage.output_tokens"]
        + anthropic_span.attributes["gen_ai.usage.input_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify request and inputs
    assert (
        anthropic_span.attributes["gen_ai.prompt.0.content"]
        == "What is the weather like right now in New York? Also what time is it there now?"
    )
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["llm.request.functions.0.name"] == "get_weather"
    assert (
        anthropic_span.attributes["llm.request.functions.0.description"]
        == "Get the current weather in a given location"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.0.input_schema"
    ] == json.dumps(
        {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                },
            },
            "required": ["location"],
        }
    )
    assert anthropic_span.attributes["llm.request.functions.1.name"] == "get_time"
    assert (
        anthropic_span.attributes["llm.request.functions.1.description"]
        == "Get the current time in a given time zone"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.1.input_schema"
    ] == json.dumps(
        {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        }
    )

    # verify response and output
    assert (
        anthropic_span.attributes["gen_ai.completion.0.finish_reason"]
        == response.stop_reason
    )
    assert (
        anthropic_span.attributes["gen_ai.completion.0.content"]
        == response.content[0].text
    )
    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert (
        (anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.id"])
        == response.content[1].id
    )
    assert (
        (anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.name"])
        == response.content[1].name
    )
    response_input = json.dumps(response.content[1].input)
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.arguments"]
        == response_input
    )

    assert (
        (anthropic_span.attributes["gen_ai.completion.0.tool_calls.1.id"])
        == response.content[2].id
    )
    assert (
        (anthropic_span.attributes["gen_ai.completion.0.tool_calls.1.name"])
        == response.content[2].name
    )
    response_input = json.dumps(response.content[2].input)
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.1.arguments"]
        == response_input
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01RBkXFe9TmDNNWThMz2HmGt"
    )

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_anthropic_tools_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    user_message = {
        "role": "user",
        "content": "What is the weather like right now in New York? Also what time is it there now?",
    }

    anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=TOOLS,
        messages=[user_message],
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass
    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 514
    assert (
        anthropic_span.attributes["gen_ai.usage.output_tokens"]
        + anthropic_span.attributes["gen_ai.usage.input_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 5

    # Validate user message
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the tool messages input vent
    assert_message_in_logs(
        logs[1], "gen_ai.user.message", {"content": {"tools": TOOLS}}
    )

    # Validate the ai response
    ideal_response = {
        "index": 0,
        "finish_reason": "tool_use",
        "message": {
            "content": "Certainly! I'd be happy to help you with both the current weather in New York and the current "
            "time there. Let's use the available tools to get this information for you."
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", ideal_response)

    # Validate the first tool call
    tool_call_0 = {
        "index": 1,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_012r6TBCWjRHG71j6zruYyUL",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "New York, NY", "unit": "fahrenheit"},
                },
            }
        ],
        "message": {"content": None},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", tool_call_0)

    # Validate the second tool call
    tool_call_1 = {
        "index": 2,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_01SkeBKkLCNYWNuivqFerGDd",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": {"timezone": "America/New_York"},
                },
            }
        ],
        "message": {"content": None},
    }
    assert_message_in_logs(logs[4], "gen_ai.choice", tool_call_1)


@pytest.mark.vcr
def test_anthropic_tools_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {
                "role": "user",
                "content": "What is the weather like right now in New York? Also what time is it there now?",
            }
        ],
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 514
    assert (
        anthropic_span.attributes["gen_ai.usage.output_tokens"]
        + anthropic_span.attributes["gen_ai.usage.input_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 5

    # Validate user message
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the first tool message

    assert_message_in_logs(logs[1], "gen_ai.user.message", {})

    # Validate the ai response
    ideal_response = {
        "index": 0,
        "finish_reason": "tool_use",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", ideal_response)

    # Validate the first tool call
    tool_call_0 = {
        "index": 1,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_012r6TBCWjRHG71j6zruYyUL",
                "type": "function",
                "function": {
                    "name": "get_weather",
                },
            }
        ],
        "message": {},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", tool_call_0)

    # Validate the second tool call
    tool_call_1 = {
        "index": 2,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_01SkeBKkLCNYWNuivqFerGDd",
                "type": "function",
                "function": {
                    "name": "get_time",
                },
            }
        ],
        "message": {},
    }
    assert_message_in_logs(logs[4], "gen_ai.choice", tool_call_1)


@pytest.mark.vcr
def test_anthropic_tools_history_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    response = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {
                "role": "user",
                "content": "What is the weather and current time in San Francisco?",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'll help you get the weather and current time in San Francisco.",
                    },
                    {
                        "id": "call_1",
                        "type": "tool_use",
                        "name": "get_weather",
                        "input": {"location": "San Francisco, CA"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "Sunny and 65 degrees Fahrenheit",
                        "tool_use_id": "call_1",
                    }
                ],
            },
        ],
    )

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert (
        anthropic_span.attributes["gen_ai.usage.output_tokens"]
        + anthropic_span.attributes["gen_ai.usage.input_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-haiku-20241022")

    # verify request and inputs
    assert (
        anthropic_span.attributes["gen_ai.prompt.0.content"]
        == "What is the weather and current time in San Francisco?"
    )
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    prompt_1_content = json.loads(anthropic_span.attributes["gen_ai.prompt.1.content"])
    assert prompt_1_content[0]["text"] == "I'll help you get the weather and current time in San Francisco."
    assert anthropic_span.attributes["gen_ai.prompt.1.role"] == "assistant"
    assert json.loads(anthropic_span.attributes["gen_ai.prompt.2.content"]) == [
        {
            "type": "tool_result",
            "content": "Sunny and 65 degrees Fahrenheit",
            "tool_use_id": "call_1",
        }
    ]
    assert anthropic_span.attributes["gen_ai.prompt.2.role"] == "user"
    assert anthropic_span.attributes["llm.request.functions.0.name"] == "get_weather"
    assert (
        anthropic_span.attributes["llm.request.functions.0.description"]
        == "Get the current weather in a given location"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.0.input_schema"
    ] == json.dumps(
        {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                },
            },
            "required": ["location"],
        }
    )
    assert anthropic_span.attributes["llm.request.functions.1.name"] == "get_time"
    assert (
        anthropic_span.attributes["llm.request.functions.1.description"]
        == "Get the current time in a given time zone"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.1.input_schema"
    ] == json.dumps(
        {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        }
    )

    # verify response and output
    assert (
        anthropic_span.attributes["gen_ai.completion.0.finish_reason"]
        == response.stop_reason
    )
    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.id"]
    ) == response.content[0].id
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.name"]
    ) == response.content[0].name
    response_input = json.dumps(response.content[0].input)
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.arguments"]
        == response_input
    )

    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01QJDheQSo4hSrxgtLpEJFkA"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_tools_history_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    user_message = {
        "role": "user",
        "content": "What is the weather and current time in San Francisco?",
    }
    first_assistant_message = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I'll help you get the weather and current time in San Francisco.",
            },
            {
                "id": "call_1",
                "type": "tool_use",
                "name": "get_weather",
                "input": {"location": "San Francisco, CA"},
            },
        ],
    }
    tool_result_message = {
        "role": "user",
        "content": [
            {"type": "tool_result", "content": "Sunny and 65 degrees Fahrenheit", "tool_use_id": "call_1"},
        ],
    }
    anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            user_message,
            first_assistant_message,
            tool_result_message,
        ],
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert (
        anthropic_span.attributes["gen_ai.usage.output_tokens"]
        + anthropic_span.attributes["gen_ai.usage.input_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-haiku-20241022")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 5

    # Validate user message
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    first_assistant_message.pop("role", None)
    # Validate the tool messages input event
    assert_message_in_logs(
        logs[1], "gen_ai.assistant.message", first_assistant_message
    )

    tool_result_message.pop("role", None)
    assert_message_in_logs(logs[2], "gen_ai.user.message", tool_result_message)

    assert_message_in_logs(logs[3], "gen_ai.user.message", {"content": {"tools": TOOLS}})

    # Validate the second tool call
    tool_call = {
        "index": 0,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_013CVavAKjSN7RZoE2ZN4xQJ",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": {"timezone": "America/Los_Angeles"},
                },
            }
        ],
        "message": {"content": None},
    }
    assert_message_in_logs(logs[4], "gen_ai.choice", tool_call)


@pytest.mark.vcr
def test_anthropic_tools_history_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    user_message = {
        "role": "user",
        "content": "What is the weather and current time in San Francisco?",
    }
    first_assistant_message = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I'll help you get the weather and current time in San Francisco.",
            },
            {
                "id": "call_1",
                "type": "tool_use",
                "name": "get_weather",
                "input": {"location": "San Francisco, CA"},
            },
        ],
    }
    tool_result_message = {
        "role": "user",
        "content": [
            {"type": "tool_result", "content": "Sunny and 65 degrees Fahrenheit", "tool_use_id": "call_1"},
        ],
    }
    anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            user_message,
            first_assistant_message,
            tool_result_message,
        ],
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert (
        anthropic_span.attributes["gen_ai.usage.output_tokens"]
        + anthropic_span.attributes["gen_ai.usage.input_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-haiku-20241022")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 5

    # Validate user message
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate the tool messages input event
    assert_message_in_logs(
        logs[1], "gen_ai.assistant.message", {}
    )

    assert_message_in_logs(logs[2], "gen_ai.user.message", {})

    assert_message_in_logs(logs[3], "gen_ai.user.message", {})

    # Validate the second tool call
    tool_call = {
        "index": 0,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_01S4zmdHhnEuStnkkoyVdiv6",
                "type": "function",
                "function": {
                    "name": "get_time",
                    # no arguments
                },
            }
        ],
        # empty message
        "message": {},
    }
    assert_message_in_logs(logs[4], "gen_ai.choice", tool_call)


@pytest.mark.vcr
def test_anthropic_tools_streaming_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {
                "role": "user",
                "content": "What is the weather and current time in San Francisco?",
            }
        ],
        stream=True,
    )

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    # consume the streaming iterator
    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert (
        anthropic_span.attributes["gen_ai.usage.output_tokens"]
        + anthropic_span.attributes["gen_ai.usage.input_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    # verify request and inputs
    assert (
        anthropic_span.attributes["gen_ai.prompt.0.content"]
        == "What is the weather and current time in San Francisco?"
    )
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["llm.request.functions.0.name"] == "get_weather"
    assert (
        anthropic_span.attributes["llm.request.functions.0.description"]
        == "Get the current weather in a given location"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.0.input_schema"
    ] == json.dumps(
        {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                },
            },
            "required": ["location"],
        }
    )
    assert anthropic_span.attributes["llm.request.functions.1.name"] == "get_time"
    assert (
        anthropic_span.attributes["llm.request.functions.1.description"]
        == "Get the current time in a given time zone"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.1.input_schema"
    ] == json.dumps(
        {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        }
    )

    # verify response and output
    assert (
        anthropic_span.attributes["gen_ai.completion.0.content"]
        == "Certainly! I can help you with that information. "
        "To get the weather and current time in San Francisco, I'll need to use "
        "two separate functions. Let me fetch that data for you."
    )
    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.id"]
    ) == "toolu_0121kXsENLvoDZ72LCuAnCCz"
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.name"]
    ) == "get_time"
    assert json.loads(
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.arguments"]
    ) == {"timezone": "America/Los_Angeles"}

    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_0138UNF3YbNp49KkqZtUBWqz"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_tools_streaming_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    user_message = {
        "role": "user",
        "content": "What is the weather like right now in New York? Also what time is it there now?",
    }

    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=TOOLS,
        messages=[user_message],
        stream=True,
    )

    # consume the streaming iterator
    for _ in response:
        pass

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass
    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert (
        anthropic_span.attributes["gen_ai.usage.output_tokens"]
        + anthropic_span.attributes["gen_ai.usage.input_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 5

    # Validate user message
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the tool messages input vent
    assert_message_in_logs(
        logs[1], "gen_ai.user.message", {"content": {"tools": TOOLS}}
    )

    # Validate the ai response
    ideal_response = {
        "index": 0,
        "finish_reason": "tool_use",
        "message": {
            "content": {
                "content": "Certainly! I'd be happy to help you with both the current "
                "weather in New York and the current "
                "time there. To get this information, I'll need to use two different "
                "tools. Let's start with the weather, "
                "and then we'll check the time.",
                "type": "text",
            }
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", ideal_response)

    # Validate the first tool call
    tool_call_0 = {
        "index": 1,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_01UGYEgvuRFeXbTZKyDyqo9P",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "New York, NY", "unit": "fahrenheit"},
                },
            }
        ],
        "message": {"content": None},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", tool_call_0)

    # Validate the second tool call
    tool_call_1 = {
        "index": 2,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_01VCGwdaiXbGQJHRCzoWgK2U",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": {"timezone": "America/New_York"},
                },
            }
        ],
        "message": {"content": None},
    }
    assert_message_in_logs(logs[4], "gen_ai.choice", tool_call_1)


@pytest.mark.vcr
def test_anthropic_tools_streaming_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    user_message = {
        "role": "user",
        "content": "What is the weather like right now in New York? Also what time is it there now?",
    }

    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=TOOLS,
        messages=[user_message],
        stream=True,
    )

    # consume the streaming iterator
    for _ in response:
        pass

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass
    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert (
        anthropic_span.attributes["gen_ai.usage.output_tokens"]
        + anthropic_span.attributes["gen_ai.usage.input_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 5

    # Validate user message
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate the tool messages input vent
    assert_message_in_logs(logs[1], "gen_ai.user.message", {})

    assert_message_in_logs(logs[2], "gen_ai.choice", {
        "finish_reason": "tool_use",
        "index": 0,
        "message": {},
    })

    # Validate the first tool call
    tool_call_0 = {
        "index": 1,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_01YUs66wivdF51ENZFX8gX9S",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    # no arguments
                },
            }
        ],
        # empty message
        "message": {},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", tool_call_0)

    # Validate the second tool call
    tool_call_1 = {
        "index": 2,
        "finish_reason": "tool_use",
        "tool_calls": [
            {
                "id": "toolu_01NRtod2L7M7TBDj9GCzsZCx",
                "type": "function",
                "function": {
                    "name": "get_time",
                    # no arguments
                },
            }
        ],
        # empty message
        "message": {},
    }
    assert_message_in_logs(logs[4], "gen_ai.choice", tool_call_1)


@pytest.mark.vcr
def test_with_asyncio_run_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter
):
    asyncio.run(
        async_anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": "You help generate concise summaries of news articles and blog posts that user sends you.",
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco?",
                },
            ],
        )
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_with_asyncio_run_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter
):
    system_message = {
        "type": "text",
        "text": "You help generate concise summaries of news articles and blog posts that user sends you.",
    }

    user_message = {
        "role": "user",
        "content": "What is the weather in San Francisco?",
    }

    asyncio.run(
        async_anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=[system_message],
            messages=[user_message],
        )
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    system_message_log = logs[0]
    assert_message_in_logs(
        system_message_log,
        "gen_ai.system.message",
        {"content": [system_message]},
    )

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[1], "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": "I apologize, but I don't have access to real-time weather information. As an AI language "
            "model, I don't have the ability to check current weather conditions or forecasts. To get accurate and "
            "up-to-date weather information for San Francisco, you would need to check a reliable weather website, "
            "app, or local news source. These sources can provide you with current conditions, forecasts, and other "
            "weather-related details for specific locations."
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_with_asyncio_run_with_events_with_no_content(
    instrument_with_no_content, async_anthropic_client, span_exporter, log_exporter
):
    asyncio.run(
        async_anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": "You help generate concise summaries of news articles and blog posts that user sends you.",
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco?",
                },
            ],
        )
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    system_message_log = logs[0]
    assert_message_in_logs(system_message_log, "gen_ai.system.message", {})

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


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.ANTHROPIC.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


@pytest.mark.vcr
def test_anthropic_message_stream_manager_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    response_content = ""
    with anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01MCkQZZtEKF3nVbFaExwATe"
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(resource_metrics, "claude-3-5-haiku-20241022")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_anthropic_message_stream_manager_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    user_message = {
        "role": "user",
        "content": "Tell me a joke about OpenTelemetry",
    }

    response_content = ""
    with anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[user_message],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(resource_metrics, "claude-3-5-haiku-20241022")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "text",
                "content": response_content,
            }
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_message_stream_manager_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    response_content = ""
    with anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(resource_metrics, "claude-3-5-haiku-20241022")

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
@pytest.mark.asyncio
async def test_async_anthropic_message_stream_manager_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter, reader
):
    response_content = ""
    async with async_anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01E414PCSTg6skd6JWPTX5Uc"
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-haiku-20241022")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_stream_manager_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter, reader
):
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    user_message = {
        "role": "user",
        "content": "Tell me a joke about OpenTelemetry",
    }

    response_content = ""
    async with async_anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[user_message],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-haiku-20241022")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": {"type": "text", "content": response_content}},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_stream_manager_with_events_with_no_content(
    instrument_with_no_content,
    async_anthropic_client,
    span_exporter,
    log_exporter,
    reader,
):
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response_content = ""
    async with async_anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 17
    assert (
        anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        + anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-haiku-20241022")

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


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_anthropic_streaming_helper_methods_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter, reader
):
    """Test that streaming helper methods like get_final_message() work with instrumentation"""
    # Test async stream with get_final_message
    async with async_anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Say hello there!",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        # Test that get_final_message() actually works (this is the main fix verification)
        message = await stream.get_final_message()
        assert message is not None
        assert hasattr(message, 'content')
        assert len(message.content) > 0
        # Test that the stream still has other helper methods available
        assert hasattr(stream, 'text_stream')
        assert hasattr(stream, 'until_done')

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    assert spans[0].name == "anthropic.chat"


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_anthropic_text_stream_helper_method_legacy(
    instrument_legacy, async_anthropic_client, span_exporter
):
    """Test that text_stream() helper method works with instrumentation"""
    async with async_anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Say hello there!",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        # Test that text_stream() works
        text_content = ""
        async for text in stream.text_stream:
            text_content += text
        assert len(text_content) > 0
    spans = span_exporter.get_finished_spans()
    print(f"Number of spans created: {len(spans)}")
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    assert spans[0].name == "anthropic.chat"


@pytest.mark.vcr()
def test_anthropic_sync_streaming_helper_methods_legacy(
    instrument_legacy, anthropic_client, span_exporter
):
    """Test that sync streaming helper methods work with instrumentation"""
    # Test sync stream - this should work similarly without helper methods causing issues
    with anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Say hello there!",
            }
        ],
        model="claude-3-5-haiku-20241022",
    ) as stream:
        # Collect all events
        events = []
        for event in stream:
            events.append(event)
        assert len(events) > 0
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "anthropic.chat"
