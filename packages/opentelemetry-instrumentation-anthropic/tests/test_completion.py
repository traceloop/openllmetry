import json
import pytest
from pathlib import Path
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT, AsyncAnthropic


@pytest.mark.vcr
def test_anthropic_completion(exporter):
    client = Anthropic()
    client.completions.create(
        prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.completion",
    ]
    anthropic_span = spans[0]
    assert (
            anthropic_span.attributes["llm.prompts.0.user"]
            == f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    )
    assert anthropic_span.attributes.get("llm.completions.0.content")


@pytest.mark.vcr
def test_anthropic_message_create(exporter):
    client = Anthropic()
    response = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.completion",
    ]
    anthropic_span = spans[0]
    assert (
            anthropic_span.attributes["llm.prompts.0.user"]
            == "Tell me a joke about OpenTelemetry"
    )
    assert (
            anthropic_span.attributes.get("llm.completions.0.content")
            == response.content[0].text
    )
    assert anthropic_span.attributes["llm.usage.prompt_tokens"] == 8
    assert (
            anthropic_span.attributes["llm.usage.completion_tokens"]
            + anthropic_span.attributes["llm.usage.prompt_tokens"]
            == anthropic_span.attributes["llm.usage.total_tokens"]
    )


@pytest.mark.vcr
def test_anthropic_multi_modal(exporter):
    client = Anthropic()
    response = client.messages.create(
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
                            "data": Path(__file__).parent.joinpath("data/logo.jpg"),
                        },
                    },
                ],
            },
        ],
        model="claude-3-opus-20240229",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.completion",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes["llm.prompts.0.user"] == json.dumps(
        [
            {"type": "text", "text": "What do you see?"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": str(Path(__file__).parent.joinpath("data/logo.jpg")),
                },
            },
        ]
    )
    assert (
            anthropic_span.attributes.get("llm.completions.0.content")
            == response.content[0].text
    )
    assert anthropic_span.attributes["llm.usage.prompt_tokens"] == 1381
    assert (
            anthropic_span.attributes["llm.usage.completion_tokens"]
            + anthropic_span.attributes["llm.usage.prompt_tokens"]
            == anthropic_span.attributes["llm.usage.total_tokens"]
    )


@pytest.mark.vcr
def test_anthropic_message_streaming(exporter):
    client = Anthropic()
    response = client.messages.create(
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
    for event in response:
        if event.type == 'content_block_delta' and event.delta.type == 'text_delta':
            response_content += event.delta.text

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.completion",
    ]
    anthropic_span = spans[0]
    assert (
            anthropic_span.attributes["llm.prompts.0.user"]
            == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes.get("llm.completions.0.content") == response_content)
    assert anthropic_span.attributes["llm.usage.prompt_tokens"] == 8
    assert (
            anthropic_span.attributes["llm.usage.completion_tokens"]
            + anthropic_span.attributes["llm.usage.prompt_tokens"]
            == anthropic_span.attributes["llm.usage.total_tokens"]
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_create(exporter):
    client = AsyncAnthropic()
    response = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.completion",
    ]
    anthropic_span = spans[0]
    assert (
            anthropic_span.attributes["llm.prompts.0.user"]
            == "Tell me a joke about OpenTelemetry"
    )
    assert (
            anthropic_span.attributes.get("llm.completions.0.content")
            == response.content[0].text
    )
    assert anthropic_span.attributes["llm.usage.prompt_tokens"] == 8
    assert (
            anthropic_span.attributes["llm.usage.completion_tokens"]
            + anthropic_span.attributes["llm.usage.prompt_tokens"]
            == anthropic_span.attributes["llm.usage.total_tokens"]
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_streaming(exporter):
    client = AsyncAnthropic()
    response = await client.messages.create(
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
        if event.type == 'content_block_delta' and event.delta.type == 'text_delta':
            response_content += event.delta.text

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.completion",
    ]
    anthropic_span = spans[0]
    assert (
            anthropic_span.attributes["llm.prompts.0.user"]
            == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes.get("llm.completions.0.content") == response_content)
    assert anthropic_span.attributes["llm.usage.prompt_tokens"] == 8
    assert (
            anthropic_span.attributes["llm.usage.completion_tokens"]
            + anthropic_span.attributes["llm.usage.prompt_tokens"]
            == anthropic_span.attributes["llm.usage.total_tokens"]
    )
