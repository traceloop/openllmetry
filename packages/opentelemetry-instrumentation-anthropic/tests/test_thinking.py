import pytest
from anthropic import Anthropic, AsyncAnthropic

from .utils import verify_metrics


@pytest.mark.vcr
def test_anthropic_thinking(exporter, reader):
    client = Anthropic()

    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    spans = exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert anthropic_span.attributes["gen_ai.completion.0.content"] == response.content[0].thinking

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert anthropic_span.attributes["gen_ai.completion.1.content"] == response.content[1].text

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking(exporter, reader):
    client = AsyncAnthropic()

    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = await client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    spans = exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert anthropic_span.attributes["gen_ai.completion.0.content"] == response.content[0].thinking

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert anthropic_span.attributes["gen_ai.completion.1.content"] == response.content[1].text

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")


@pytest.mark.vcr
def test_anthropic_thinking_streaming(exporter, reader):
    client = Anthropic()

    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        stream=True,
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    text = ""
    thinking = ""

    for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            text += event.delta.text
        elif event.type == "content_block_delta" and event.delta.type == "thinking_delta":
            thinking += event.delta.thinking

    spans = exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert anthropic_span.attributes["gen_ai.completion.0.content"] == thinking

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert anthropic_span.attributes["gen_ai.completion.1.content"] == text

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_streaming(exporter, reader):
    client = AsyncAnthropic()

    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = await client.messages.create(
        model="claude-3-7-sonnet-20250219",
        stream=True,
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    text = ""
    thinking = ""

    async for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            text += event.delta.text
        elif event.type == "content_block_delta" and event.delta.type == "thinking_delta":
            thinking += event.delta.thinking

    spans = exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert anthropic_span.attributes["gen_ai.completion.0.content"] == thinking

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert anthropic_span.attributes["gen_ai.completion.1.content"] == text

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")
