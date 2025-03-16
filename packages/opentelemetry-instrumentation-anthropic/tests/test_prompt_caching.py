from pathlib import Path

import pytest
from anthropic import Anthropic, AsyncAnthropic

from .utils import verify_metrics


@pytest.mark.vcr
def test_anthropic_prompt_caching(exporter, reader):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = "test_anthropic_prompt_caching <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n" + f.read()
    client = Anthropic()

    try:
        client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
        )

    spans = exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_creation_span.attributes["gen_ai.prompt.0.content"]
    assert cache_read_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_read_span.attributes["gen_ai.prompt.0.content"]

    assert cache_creation_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_creation_span.attributes["gen_ai.prompt.1.content"]
    assert cache_read_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_read_span.attributes["gen_ai.prompt.1.content"]
    assert (
        cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"]
        == cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"]
    )
    assert cache_creation_span.attributes.get("gen_ai.response.id") == "msg_01EF3r8zYyZntM4Sg9a5kc6k"
    assert cache_read_span.attributes.get("gen_ai.response.id") == "msg_01YGB3PuEANUSkLuzemhtNVF"

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    # first check that cache_creation_span only wrote to cache, but not read from it,
    assert cache_creation_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 0
    assert cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"] != 0

    # then check for exact figures for the fixture/cassete
    assert cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 1163
    assert cache_creation_span.attributes["gen_ai.usage.prompt_tokens"] == 1167
    assert cache_creation_span.attributes["gen_ai.usage.completion_tokens"] == 187

    # first check that cache_read_span only read from cache, but not wrote to it,
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] != 0
    assert cache_read_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 0

    # then check for exact figures for the fixture/cassete
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 1163
    assert cache_read_span.attributes["gen_ai.usage.prompt_tokens"] == 1167
    assert cache_read_span.attributes["gen_ai.usage.completion_tokens"] == 202

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_prompt_caching_async(exporter, reader):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = "test_anthropic_prompt_caching_async <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n" + f.read()
    client = AsyncAnthropic()

    try:
        await client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        await client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
        )

    spans = exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_creation_span.attributes["gen_ai.prompt.0.content"]
    assert cache_read_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_read_span.attributes["gen_ai.prompt.0.content"]

    assert cache_creation_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_creation_span.attributes["gen_ai.prompt.1.content"]
    assert cache_read_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_read_span.attributes["gen_ai.prompt.1.content"]
    assert (
        cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"]
        == cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"]
    )
    assert cache_creation_span.attributes.get("gen_ai.response.id") == "msg_01AGcJaUoaQe4VfWUjnSBrXg"
    assert cache_read_span.attributes.get("gen_ai.response.id") == "msg_01Q8hYZvCMAQKC4n8X3zFnrX"

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    # first check that cache_creation_span only wrote to cache, but not read from it,
    assert cache_creation_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 0
    assert cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"] != 0

    # then check for exact figures for the fixture/cassete
    assert cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 1165
    assert cache_creation_span.attributes["gen_ai.usage.prompt_tokens"] == 1169
    assert cache_creation_span.attributes["gen_ai.usage.completion_tokens"] == 207

    # first check that cache_read_span only read from cache, but not wrote to it,
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] != 0
    assert cache_read_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 0

    # then check for exact figures for the fixture/cassete
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 1165
    assert cache_read_span.attributes["gen_ai.usage.prompt_tokens"] == 1169
    assert cache_read_span.attributes["gen_ai.usage.completion_tokens"] == 224

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")


@pytest.mark.vcr
def test_anthropic_prompt_caching_stream(exporter, reader):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = "test_anthropic_prompt_caching_stream <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n" + f.read()
    client = Anthropic()
    try:
        client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            stream=True,
            system=[
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
        )
        response_content = ""
        for event in response:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_creation_span.attributes["gen_ai.prompt.0.content"]
    assert cache_read_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_read_span.attributes["gen_ai.prompt.0.content"]

    assert cache_creation_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_creation_span.attributes["gen_ai.prompt.1.content"]
    assert cache_read_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_read_span.attributes["gen_ai.prompt.1.content"]
    assert (
        cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"]
        == cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"]
    )
    assert cache_creation_span.attributes.get("gen_ai.response.id") == "msg_017FfRkh9PCC8YbjnhDMrPuK"
    assert cache_read_span.attributes.get("gen_ai.response.id") == "msg_01XQRA3bs4SB4yTBMwD3dbUi"

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    # first check that cache_creation_span only wrote to cache, but not read from it,
    assert cache_creation_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 0
    assert cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"] != 0

    # then check for exact figures for the fixture/cassete
    assert cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 1165
    assert cache_creation_span.attributes["gen_ai.usage.prompt_tokens"] == 1169
    assert cache_creation_span.attributes["gen_ai.usage.completion_tokens"] == 202

    # first check that cache_read_span only read from cache, but not wrote to it,
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] != 0
    assert cache_read_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 0

    # then check for exact figures for the fixture/cassete
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 1165
    assert cache_read_span.attributes["gen_ai.usage.prompt_tokens"] == 1169
    assert cache_read_span.attributes["gen_ai.usage.completion_tokens"] == 222

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_prompt_caching_async_stream(exporter, reader):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = "test_anthropic_prompt_caching_async_stream <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n" + f.read()
    client = AsyncAnthropic()
    try:
        await client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        response = await client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            stream=True,
            system=[
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
        )
        response_content = ""
        async for event in response:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_creation_span.attributes["gen_ai.prompt.0.content"]
    assert cache_read_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_read_span.attributes["gen_ai.prompt.0.content"]
    assert cache_creation_span.attributes.get("gen_ai.response.id") == "msg_01KQCu5jXyou55u6YFNk6uqu"
    assert cache_read_span.attributes.get("gen_ai.response.id") == "msg_01GZo7EAMfEuzRqTKrFANNpA"

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert cache_creation_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_creation_span.attributes["gen_ai.prompt.1.content"]
    assert cache_read_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_read_span.attributes["gen_ai.prompt.1.content"]
    assert (
        cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"]
        == cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"]
    )

    # first check that cache_creation_span only wrote to cache, but not read from it,
    assert cache_creation_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 0
    assert cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"] != 0

    # then check for exact figures for the fixture/cassete
    assert cache_creation_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 1167
    assert cache_creation_span.attributes["gen_ai.usage.prompt_tokens"] == 1171
    assert cache_creation_span.attributes["gen_ai.usage.completion_tokens"] == 290

    # first check that cache_read_span only read from cache, but not wrote to it,
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] != 0
    assert cache_read_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 0

    # then check for exact figures for the fixture/cassete
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 1167
    assert cache_read_span.attributes["gen_ai.usage.prompt_tokens"] == 1171
    assert cache_read_span.attributes["gen_ai.usage.completion_tokens"] == 257

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")
