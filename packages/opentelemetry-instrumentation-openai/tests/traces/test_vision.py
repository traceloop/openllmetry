import json
import pytest


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
    assert json.loads(open_ai_span.attributes["gen_ai.prompt.0.content"]) == [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://source.unsplash.com/8xznAGy4HcY/800x400"},
        },
    ]

    assert open_ai_span.attributes.get("gen_ai.completion.0.content")
    assert open_ai_span.attributes["openai.api_base"] == "https://api.openai.com/v1/"
