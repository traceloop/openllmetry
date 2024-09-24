import json
import pytest
import base64
import random
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.instrumentation.openai.shared.chat_wrappers import _process_image_item, Config


@pytest.mark.vcr
def test_vision(exporter, openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
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

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert json.loads(
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    ) == [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://source.unsplash.com/8xznAGy4HcY/800x400"},
        },
    ]

    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )


def test_process_image_item_with_base64_upload(mock_upload_base64_image):
    trace_id = "test_trace_id"
    span_id = "test_span_id"
    message_index = 0
    content_index = 1
    base64_image = base64.b64encode(b"fake image data").decode('utf-8')
    item = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }

    Config.upload_base64_image = mock_upload_base64_image
    result = _process_image_item(item, trace_id, span_id, message_index, content_index)
    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == "https://example.com/uploaded_image.jpg"
    
    mock_upload_base64_image.assert_called_once_with(
        trace_id,
        span_id,
        "message_0_content_1.jpeg",
        base64_image
    )

def test_process_image_item_without_base64_upload():
    trace_id = "test_trace_id"
    span_id = "test_span_id"
    message_index = 0
    content_index = 1
    base64_image = base64.b64encode(b"fake image data").decode('utf-8')
    item = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }

    Config.upload_base64_image = None
    result = _process_image_item(item, trace_id, span_id, message_index, content_index)
    assert result == item  

def test_process_image_item_non_base64():
    trace_id = "test_trace_id"
    span_id = "test_span_id"
    message_index = 0
    content_index = 1
    item = {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/image.jpg"
        }
    }

    result = _process_image_item(item, trace_id, span_id, message_index, content_index)
    assert result == item
    