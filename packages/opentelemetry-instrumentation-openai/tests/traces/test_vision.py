import base64

import pytest
import requests
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from .utils import get_input_messages, get_output_messages


@pytest.mark.vcr
def test_vision(instrument_legacy, span_exporter, log_exporter, openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://source.unsplash.com/8xznAGy4HcY/800x400"
                        },
                    },
                ],
            }
        ],
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    input_messages = get_input_messages(open_ai_span)
    assert input_messages[0]["role"] == "user"
    parts = input_messages[0]["parts"]
    text_parts = [p for p in parts if p.get("type") == "text"]
    uri_parts = [p for p in parts if p.get("type") == "uri"]
    assert text_parts[0]["content"] == "What is in this image?"
    assert len(uri_parts) == 1
    assert uri_parts[0]["uri"] == "https://source.unsplash.com/8xznAGy4HcY/800x400"
    output_messages = get_output_messages(open_ai_span)
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert (
        open_ai_span.attributes[SpanAttributes.GEN_AI_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-8wq4EsSXTQC0JbGzob3SBHg6pS7Tt"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_vision_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://source.unsplash.com/8xznAGy4HcY/800x400"
                        },
                    },
                ],
            }
        ],
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[SpanAttributes.GEN_AI_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-8wq4EsSXTQC0JbGzob3SBHg6pS7Tt"
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
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://source.unsplash.com/8xznAGy4HcY/800x400"
                    },
                },
            ]
        },
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {
            "content": response.choices[0].message.content,
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_vision_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://source.unsplash.com/8xznAGy4HcY/800x400"
                        },
                    },
                ],
            }
        ],
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[SpanAttributes.GEN_AI_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-8wq4EsSXTQC0JbGzob3SBHg6pS7Tt"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "length", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_vision_base64(instrument_legacy, span_exporter, log_exporter, openai_client):
    # Fetch the image from the URL
    response = requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/"
        "thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
        "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )
    image_data = response.content

    # Encode the image data to base64
    base64_image = base64.b64encode(image_data).decode("utf-8")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    input_messages = get_input_messages(open_ai_span)
    assert input_messages[0]["role"] == "user"
    parts = input_messages[0]["parts"]
    text_parts = [p for p in parts if p.get("type") == "text"]
    uri_parts = [p for p in parts if p.get("type") == "uri"]
    assert text_parts[0]["content"] == "What is in this image?"
    assert len(uri_parts) == 1
    # base64 images are replaced with /some/url
    assert uri_parts[0]["uri"] == "/some/url"
    output_messages = get_output_messages(open_ai_span)
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert (
        open_ai_span.attributes[SpanAttributes.GEN_AI_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AC7YAG2uy8c4VfbqJp4QkdHc5PDZ4"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_vision_base64_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    # Fetch the image from the URL
    response = requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/"
        "thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
        "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )
    image_data = response.content

    # Encode the image data to base64
    base64_image = base64.b64encode(image_data).decode("utf-8")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[SpanAttributes.GEN_AI_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AC7YAG2uy8c4VfbqJp4QkdHc5PDZ4"
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
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        },
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": response.choices[0].message.content,
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_vision_base64_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    # Fetch the image from the URL
    response = requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/"
        "thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
        "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )
    image_data = response.content

    # Encode the image data to base64
    base64_image = base64.b64encode(image_data).decode("utf-8")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[SpanAttributes.GEN_AI_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AC7YAG2uy8c4VfbqJp4QkdHc5PDZ4"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == GenAIAttributes.GenAiSystemValues.OPENAI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


@pytest.mark.asyncio
async def test_vision_default_uploader_config():
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.instrumentation.openai.shared.config import Config

    original = Config.upload_base64_image
    try:
        OpenAIInstrumentor()
        assert Config.upload_base64_image is None
    finally:
        Config.upload_base64_image = original


@pytest.mark.asyncio
async def test_process_image_item_uploader_handling():
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
        _process_image_item,
    )
    from opentelemetry.instrumentation.openai.shared.config import Config

    original = Config.upload_base64_image
    item = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
    }

    try:
        # 1. No uploader configured (None) -> returns original item
        Config.upload_base64_image = None
        res_none = await _process_image_item(item, "trace1", "span1", 0, 0)
        assert res_none == item

        # 2. Synchronous uploader -> returns url without TypeError
        Config.upload_base64_image = (
            lambda trace_id, span_id, name, b64: "https://cdn.example.com/sync.png"
        )
        res_sync = await _process_image_item(item, "trace1", "span1", 0, 0)
        assert res_sync == {
            "type": "image_url",
            "image_url": {"url": "https://cdn.example.com/sync.png"},
        }

        # 3. Asynchronous uploader -> awaited and returns url
        async def async_uploader(trace_id, span_id, name, b64):
            return "https://cdn.example.com/async.png"

        Config.upload_base64_image = async_uploader
        res_async = await _process_image_item(item, "trace1", "span1", 0, 0)
        assert res_async == {
            "type": "image_url",
            "image_url": {"url": "https://cdn.example.com/async.png"},
        }
    finally:
        Config.upload_base64_image = original


@pytest.mark.asyncio
async def test_set_prompts_preserves_input_messages_without_uploader(
    tracer_provider, span_exporter
):
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import _set_prompts
    from opentelemetry.instrumentation.openai.shared.config import Config

    original = Config.upload_base64_image
    tracer = tracer_provider.get_tracer("test")
    span = tracer.start_span("test_vision_no_uploader_span")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                },
            ],
        }
    ]

    try:
        Config.upload_base64_image = None
        await _set_prompts(span, messages)
        span.end()

        spans = span_exporter.get_finished_spans()
        target_span = [s for s in spans if s.name == "test_vision_no_uploader_span"][-1]
        input_messages = get_input_messages(target_span)

        assert len(input_messages) == 1
        assert input_messages[0]["role"] == "user"
        parts = input_messages[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "content": "Describe this image"}
        assert parts[1] == {
            "type": "blob",
            "modality": "image",
            "mime_type": "image/png",
            "content": "iVBORw0KGgo=",
        }
    finally:
        Config.upload_base64_image = original
