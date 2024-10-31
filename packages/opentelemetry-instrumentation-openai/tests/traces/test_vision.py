import json
import pytest
import base64
from opentelemetry.semconv_ai import SpanAttributes
import requests


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


@pytest.mark.vcr
def test_vision_base64(exporter, openai_client):
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
        model="gpt-4-vision-preview",
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
            "image_url": {"url": "/some/url"},
        },
    ]

    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
