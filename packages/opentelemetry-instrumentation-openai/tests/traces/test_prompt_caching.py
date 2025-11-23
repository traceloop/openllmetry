from pathlib import Path

import pytest
from openai import AsyncOpenAI, OpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_openai_prompt_caching(instrument_legacy, span_exporter, log_exporter):
    with open(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_openai_prompt_caching <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
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

    spans = span_exporter.get_finished_spans()
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

    assert (
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNi3xzj4EEAzo6vce1IwHwie9IRhH"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNi420iFNtIOHzy8Gq2fVS5utTus7"
    )

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1149
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 315

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1149
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 353

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_openai_prompt_caching_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    with open(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_openai_prompt_caching <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
    client = OpenAI()

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    responses = []
    for _ in range(2):
        responses.append(
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
        )

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "openai.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert (
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNi3xzj4EEAzo6vce1IwHwie9IRhH"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNi420iFNtIOHzy8Gq2fVS5utTus7"
    )

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1149
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 315
    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1149
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 353

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate the first call events
    # Validate system message Event
    system_message_event = {"content": system_message}
    assert_message_in_logs(logs[0], "gen_ai.system.message", system_message_event)

    # Validate the user message Event
    user_message_event = {"content": text}
    assert_message_in_logs(logs[1], "gen_ai.user.message", user_message_event)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": responses[0].choices[0].message.content,
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second call events
    # Validate system message Event
    assert_message_in_logs(logs[3], "gen_ai.system.message", system_message_event)

    # Validate the user message Event
    assert_message_in_logs(logs[4], "gen_ai.user.message", user_message_event)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": responses[1].choices[0].message.content,
        },
    }
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_openai_prompt_caching_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    with open(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_openai_prompt_caching <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "openai.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert (
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNi3xzj4EEAzo6vce1IwHwie9IRhH"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNi420iFNtIOHzy8Gq2fVS5utTus7"
    )

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1149
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 315
    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1149
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 353

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate the first call events
    # Validate system message Event
    system_message_event = {}
    assert_message_in_logs(logs[0], "gen_ai.system.message", system_message_event)

    # Validate the user message Event
    user_message_event = {}
    assert_message_in_logs(logs[1], "gen_ai.user.message", user_message_event)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second call events
    # Validate system message Event
    assert_message_in_logs(logs[3], "gen_ai.system.message", system_message_event)

    # Validate the user message Event
    assert_message_in_logs(logs[4], "gen_ai.user.message", user_message_event)

    # Validate the ai response
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_prompt_caching_async(
    instrument_legacy, span_exporter, log_exporter
):
    with open(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_openai_prompt_caching_async <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
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

    spans = span_exporter.get_finished_spans()
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
    assert (
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNhr79TlegaJvfSOAOH2jsPEpRHMd"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNhrEFvKSNY08Uphau5iA4InZH6jn"
    )

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1150
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 293

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1150
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 307

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_prompt_caching_async_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    with open(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_openai_prompt_caching_async <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
    client = AsyncOpenAI()

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    responses = []
    for _ in range(2):
        responses.append(
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
        )

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "openai.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert (
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNhr79TlegaJvfSOAOH2jsPEpRHMd"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNhrEFvKSNY08Uphau5iA4InZH6jn"
    )

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1150
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 293

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1150
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 307

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate the first call events
    # Validate system message Event
    system_message_event = {"content": system_message}
    assert_message_in_logs(logs[0], "gen_ai.system.message", system_message_event)

    # Validate the user message Event
    user_message_event = {"content": text}
    assert_message_in_logs(logs[1], "gen_ai.user.message", user_message_event)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": responses[0].choices[0].message.content,
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second call events
    # Validate system message Event
    assert_message_in_logs(logs[3], "gen_ai.system.message", system_message_event)

    # Validate the user message Event
    assert_message_in_logs(logs[4], "gen_ai.user.message", user_message_event)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": responses[1].choices[0].message.content,
        },
    }
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_prompt_caching_async_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    with open(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_openai_prompt_caching_async <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "openai.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert (
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNhr79TlegaJvfSOAOH2jsPEpRHMd"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-BNhrEFvKSNY08Uphau5iA4InZH6jn"
    )

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1150
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 293
    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1150
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 307

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate the first call events
    # Validate system message Event
    system_message_event = {}
    assert_message_in_logs(logs[0], "gen_ai.system.message", system_message_event)

    # Validate the user message Event
    user_message_event = {}
    assert_message_in_logs(logs[1], "gen_ai.user.message", user_message_event)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second call events
    # Validate system message Event
    assert_message_in_logs(logs[3], "gen_ai.system.message", system_message_event)

    # Validate the user message Event
    assert_message_in_logs(logs[4], "gen_ai.user.message", user_message_event)

    # Validate the ai response
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.OPENAI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
