from pathlib import Path

import pytest
from openai import OpenAI, AsyncOpenAI


@pytest.mark.vcr
def test_openai_prompt_caching(exporter):
    with open(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = "test_openai_prompt_caching <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n" + f.read()
    client = OpenAI()

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

    spans = exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "openai.chat" for span in spans)
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

    assert cache_creation_span.attributes.get("gen_ai.response.id") == "chatcmpl-BNi3xzj4EEAzo6vce1IwHwie9IRhH"
    assert cache_read_span.attributes.get("gen_ai.response.id") == "chatcmpl-BNi420iFNtIOHzy8Gq2fVS5utTus7"

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert cache_creation_span.attributes["gen_ai.usage.prompt_tokens"] == 1149
    assert cache_creation_span.attributes["gen_ai.usage.completion_tokens"] == 315
    assert cache_creation_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 0

    assert cache_read_span.attributes["gen_ai.usage.prompt_tokens"] == 1149
    assert cache_read_span.attributes["gen_ai.usage.completion_tokens"] == 353
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 1024


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_prompt_caching_async(exporter):
    with open(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = "test_openai_prompt_caching_async <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n" + f.read()
    client = AsyncOpenAI()

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

    spans = exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "openai.chat" for span in spans)
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
    assert cache_creation_span.attributes.get("gen_ai.response.id") == "chatcmpl-BNhr79TlegaJvfSOAOH2jsPEpRHMd"
    assert cache_read_span.attributes.get("gen_ai.response.id") == "chatcmpl-BNhrEFvKSNY08Uphau5iA4InZH6jn"

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert cache_creation_span.attributes["gen_ai.usage.prompt_tokens"] == 1150
    assert cache_creation_span.attributes["gen_ai.usage.completion_tokens"] == 293
    assert cache_creation_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 0

    assert cache_read_span.attributes["gen_ai.usage.prompt_tokens"] == 1150
    assert cache_read_span.attributes["gen_ai.usage.completion_tokens"] == 307
    assert cache_read_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 1024
