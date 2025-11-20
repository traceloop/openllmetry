from pathlib import Path

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .utils import verify_metrics


@pytest.mark.vcr
def test_anthropic_prompt_caching_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
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
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "msg_01EF3r8zYyZntM4Sg9a5kc6k"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "msg_01YGB3PuEANUSkLuzemhtNVF"
    )

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    # assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1167
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 187

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1167
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 202

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_prompt_caching_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1167
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 187

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1167
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 202

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate the first system message Event
    system_message_log = logs[0]
    assert_message_in_logs(
        system_message_log,
        "gen_ai.system.message",
        {
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
        },
    )

    # Validate the first user message Event
    user_message_log = logs[1]
    ideal_user_log_message = {
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", ideal_user_log_message
    )

    # Validate the first the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": "Here are concise summaries of the three articles:\n\n1. OpenLLMetry: New open-source library "
            "extends OpenTelemetry with LLM functionality, enabling developers to monitor AI model performance in "
            "applications. Key features include LLM-specific metrics/traces and integration with existing setups.\n\n"
            "2. Major LLM providers introduce prompt caching, dramatically improving speed and reducing costs for API "
            "calls. Benefits include millisecond response times, lower computational resources, and improved "
            "scalability. Particularly useful for applications with repetitive prompts like chatbots and content "
            "moderation.\n\n3. Unit testing is crucial in software development: It catches bugs early, saves "
            "time/money, improves code quality, facilitates refactoring, serves as documentation, and enhances "
            "collaboration. The post emphasizes that unit testing is a professional responsibility that pays off in "
            "the long run."
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second system message Event
    system_message_log = logs[3]
    assert_message_in_logs(
        system_message_log,
        "gen_ai.system.message",
        {
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
        },
    )

    # Validate the second user message Event
    user_message_log = logs[4]
    ideal_user_log_message = {
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", ideal_user_log_message
    )

    # Validate the second the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": "Here are concise summaries of the three articles:\n\n1. OpenLLMetry: New open-source library "
            "extending OpenTelemetry with LLM functionality. Key features include LLM-specific metrics/traces, "
            "integration with existing setups, and support for popular LLM frameworks. Aims to improve monitoring of "
            "AI-powered systems.\n\n2. Major LLM providers introduce prompt caching to boost speed and reduce costs. "
            "Benefits include faster response times, lower computational costs, and improved scalability. Particularly "
            "useful for applications with repetitive prompts like chatbots and content moderation. Expected to drive "
            "wider LLM adoption and new application paradigms.\n\n3. Argument for the importance of unit testing in "
            "software development. Benefits include early bug detection, time/cost savings, improved code quality, "
            "easier refactoring, documentation, and enhanced collaboration. Author argues unit testing is a "
            "professional responsibility that pays off in the long run."
        },
    }
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_prompt_caching_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1167
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 187

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1167
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 202

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    i = 0
    for _ in range(2):
        # Validate the system message Event
        system_message_log = logs[i]
        assert_message_in_logs(system_message_log, "gen_ai.system.message", {})
        i += 1

        # Validate the user message Event
        user_message_log = logs[i]
        assert_message_in_logs(user_message_log, "gen_ai.user.message", {})
        i += 1

        # Validate the the ai response
        choice_event = {
            "index": 0,
            "finish_reason": "end_turn",
            "message": {},
        }
        assert_message_in_logs(logs[i], "gen_ai.choice", choice_event)
        i += 1


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_prompt_caching_async_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching_async <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        await async_anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
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
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "msg_01AGcJaUoaQe4VfWUjnSBrXg"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "msg_01Q8hYZvCMAQKC4n8X3zFnrX"
    )

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 207

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 224

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_prompt_caching_async_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching_async <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        await async_anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 207

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 224

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate the first system message Event
    system_message_log = logs[0]
    assert_message_in_logs(
        system_message_log,
        "gen_ai.system.message",
        {
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                },
            ]
        },
    )

    # Validate the first user message Event
    user_message_log = logs[1]
    ideal_user_log_message = {
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", ideal_user_log_message
    )

    # Validate the first the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": "Here are concise summaries of the three articles:\n\n1. OpenLLMetry: New open-source library "
            "extending OpenTelemetry with LLM functionality. Developed by Traceloop, it provides LLM-specific metrics "
            "and traces, integrates with existing setups, and supports popular LLM frameworks. Aims to improve "
            "monitoring of AI-powered systems.\n\n2. Major LLM providers introduce prompt caching: New feature stores "
            "responses for frequent prompts, improving speed and reducing costs. Benefits include millisecond response "
            "times, lower computational resources, and improved scalability. Particularly useful for applications with "
            "repetitive prompts like chatbots and translation services. \n\n3. Importance of unit testing in software "
            "development: Key benefits include early bug detection, time and cost savings, improved code quality, "
            "easier refactoring, documentation, and enhanced collaboration. The author argues unit testing is crucial "
            "for professional software development and pays off in the long run."
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second system message Event
    system_message_log = logs[3]
    assert_message_in_logs(
        system_message_log,
        "gen_ai.system.message",
        {
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
        },
    )

    # Validate the second user message Event
    user_message_log = logs[4]
    ideal_user_log_message = {
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", ideal_user_log_message
    )

    # Validate the second the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": "Here are concise summaries of the three articles:\n\n1. OpenLLMetry: New open-source library "
            "extending OpenTelemetry with LLM functionality. Key features include LLM-specific metrics/traces, "
            "integration with existing setups, and support for popular LLM frameworks. Aims to provide deeper insights "
            "into AI model performance within applications.\n\n2. Major LLM providers introduce prompt caching to "
            "improve speed and reduce costs. Benefits include millisecond response times, lower computational "
            "resources, and improved scalability. Particularly useful for applications with repetitive prompts like "
            "chatbots and content moderation. Expected to impact AI industry through wider adoption and new application"
            " paradigms.\n\n3. Importance of unit testing in software development:\n- Catches bugs early\n- Saves time "
            "and money \n- Improves code quality\n- Facilitates refactoring\n- Serves as documentation\n- Enhances "
            "collaboration\nThe post emphasizes that unit testing is crucial for professional software development and "
            "pays off in the long run."
        },
    }
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_prompt_caching_async_with_events_with_no_content(
    instrument_with_no_content,
    async_anthropic_client,
    span_exporter,
    log_exporter,
    reader,
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching_async <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        await async_anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 207

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 224

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    i = 0
    for _ in range(2):
        # Validate the system message Event
        system_message_log = logs[i]
        assert_message_in_logs(system_message_log, "gen_ai.system.message", {})
        i += 1

        # Validate the user message Event
        user_message_log = logs[i]
        assert_message_in_logs(user_message_log, "gen_ai.user.message", {})
        i += 1

        # Validate the the ai response
        choice_event = {
            "index": 0,
            "finish_reason": "end_turn",
            "message": {},
        }
        assert_message_in_logs(logs[i], "gen_ai.choice", choice_event)
        i += 1


@pytest.mark.vcr
def test_anthropic_prompt_caching_stream_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching_stream <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        response = anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
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
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "msg_017FfRkh9PCC8YbjnhDMrPuK"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "msg_01XQRA3bs4SB4yTBMwD3dbUi"
    )

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 202

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 222

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_prompt_caching_stream_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching_stream <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        response = anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 202

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 222

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate the first system message Event
    system_message_log = logs[0]
    assert_message_in_logs(
        system_message_log,
        "gen_ai.system.message",
        {
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
        },
    )

    # Validate the first user message Event
    user_message_log = logs[1]
    ideal_user_log_message = {
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", ideal_user_log_message
    )

    # Validate the first the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "text",
                "content": "Here are concise summaries of the three articles:\n\n1. OpenLLMetry: New open-source "
                "library extends OpenTelemetry with LLM functionality, enabling deeper insights into AI model "
                "performance in applications. Key features include LLM-specific metrics/traces and integration with "
                "existing setups.\n\n2. Major LLM providers introduce prompt caching, dramatically improving speed and "
                "reducing costs for API calls. Benefits include millisecond response times, lower computational "
                "resources, and improved scalability. Particularly useful for applications with repetitive prompts "
                "like chatbots and content moderation.\n\n3. Unit testing is crucial in software development. Key "
                "benefits:\n- Catches bugs early\n- Saves time and money \n- Improves code quality\n- Facilitates "
                "refactoring\n- Serves as documentation\n- Enhances collaboration\nThe post emphasizes that unit "
                "testing is a professional responsibility with long-term payoffs.",
            }
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second system message Event
    system_message_log = logs[3]
    assert_message_in_logs(
        system_message_log,
        "gen_ai.system.message",
        {
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
        },
    )

    # Validate the second user message Event
    user_message_log = logs[4]
    ideal_user_log_message = {
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", ideal_user_log_message
    )

    # Validate the second the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "text",
                "content": "Here are concise summaries of the three articles:\n\n1. OpenLLMetry: New open-source "
                "library extending OpenTelemetry with LLM functionality. Key features include LLM-specific "
                "metrics/traces, integration with existing setups, and support for popular LLM frameworks. Aims to "
                "provide deeper insights into AI model performance in applications.\n\n2. Major LLM providers "
                "introduce prompt caching to improve speed and reduce costs. Benefits include faster response times, "
                "lower computational costs, and improved scalability. Particularly useful for applications with "
                "repetitive prompts like chatbots and content moderation. Expected to impact AI industry by enabling "
                "wider adoption and new application types.\n\n3. Importance of unit testing in software development:\n-"
                " Catches bugs early\n- Saves time and money\n- Improves code quality\n- Facilitates refactoring\n- "
                "Serves as documentation\n- Enhances collaboration\nThe post emphasizes that unit testing is crucial "
                "for professional software development and pays off in the long run.",
            }
        },
    }
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_prompt_caching_stream_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching_stream <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        response = anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 202

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1169
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 222

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    i = 0
    for _ in range(2):
        # Validate the system message Event
        system_message_log = logs[i]
        assert_message_in_logs(system_message_log, "gen_ai.system.message", {})
        i += 1

        # Validate the user message Event
        user_message_log = logs[i]
        assert_message_in_logs(user_message_log, "gen_ai.user.message", {})
        i += 1

        # Validate the the ai response
        choice_event = {
            "index": 0,
            "finish_reason": "end_turn",
            "message": {},
        }
        assert_message_in_logs(logs[i], "gen_ai.choice", choice_event)
        i += 1


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_prompt_caching_async_stream_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching_async_stream <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        response = await async_anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_creation_span.attributes["gen_ai.prompt.0.content"]
    assert cache_read_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert system_message == cache_read_span.attributes["gen_ai.prompt.0.content"]
    assert (
        cache_creation_span.attributes.get("gen_ai.response.id")
        == "msg_01KQCu5jXyou55u6YFNk6uqu"
    )
    assert (
        cache_read_span.attributes.get("gen_ai.response.id")
        == "msg_01GZo7EAMfEuzRqTKrFANNpA"
    )

    assert cache_creation_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert cache_read_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert cache_creation_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_creation_span.attributes["gen_ai.prompt.1.content"]
    assert cache_read_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert text == cache_read_span.attributes["gen_ai.prompt.1.content"]

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1171
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 290

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1171
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 257

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_prompt_caching_async_stream_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter, reader
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching_async_stream <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        response = await async_anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1171
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 290

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1171
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 257

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate the first system message Event
    system_message_log = logs[0]
    assert_message_in_logs(
        system_message_log,
        "gen_ai.system.message",
        {
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
        },
    )

    # Validate the first user message Event
    user_message_log = logs[1]
    ideal_user_log_message = {
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", ideal_user_log_message
    )

    # Validate the first the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "text",
                "content": "Here are concise summaries of the three articles:\n\n1. OpenLLMetry: New Open-Source "
                "Library for LLM Monitoring\n\nTraceloop released OpenLLMetry, an open-source library extending "
                "OpenTelemetry with LLM functionality. Key features:\n- LLM-specific metrics and traces\n- Integration "
                "with existing OpenTelemetry setups\n- Support for popular LLM frameworks\nAims to provide deeper "
                "insights into AI model performance in applications.\n\n2. Major LLM Providers Introduce Prompt Caching"
                "\n\nLeading LLM providers, including Anthropic, have implemented prompt caching to improve speed and "
                "reduce costs. Benefits:\n- Faster response times (milliseconds vs seconds)\n- Lower computational "
                "costs\n- Improved scalability\nParticularly useful for applications with repetitive prompts like "
                "chatbots and content moderation. Expected to drive wider LLM adoption and new application paradigms."
                "\n\n3. Importance of Unit Testing in Software Development\n\nA software professional advocates for "
                "unit testing as crucial to development:\n- Catches bugs early\n- Saves time and money\n- Improves code"
                " quality and architecture\n- Facilitates refactoring\n- Acts as documentation\n- Enhances team "
                "collaboration\nEmphasized as a professional responsibility with long-term benefits.",
            }
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second system message Event
    system_message_log = logs[3]
    assert_message_in_logs(
        system_message_log,
        "gen_ai.system.message",
        {
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                },
            ],
        },
    )

    # Validate the second user message Event
    user_message_log = logs[4]
    ideal_user_log_message = {
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", ideal_user_log_message
    )

    # Validate the second the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "text",
                "content": "Here are concise summaries of the three articles:\n\n1. OpenLLMetry: New Open-Source "
                "Library for LLM Monitoring\n\nTraceloop released OpenLLMetry, an open-source library extending "
                "OpenTelemetry with LLM functionality. Key features include LLM-specific metrics and traces, "
                "integration with existing setups, and support for popular LLM frameworks. It aims to provide deeper "
                "insights into AI model performance within applications.\n\n2. Major LLM Providers Introduce Prompt "
                "Caching\n\nLeading LLM providers, including Anthropic, have implemented prompt caching to improve "
                "speed and reduce costs of API calls. This technique stores responses for frequent prompts, "
                "dramatically reducing response times and computational resources. Benefits include improved "
                "scalability and cost-effectiveness, particularly for applications with repetitive prompt patterns.\n\n"
                "3. Importance of Unit Testing in Software Development\n\nThe article emphasizes the critical role of "
                "unit testing in software development. Key benefits include early bug detection, time and cost savings,"
                " improved code quality, easier refactoring, code documentation, and enhanced team collaboration. The "
                "author argues that unit testing is a professional responsibility that pays off significantly in the "
                "long run.",
            }
        },
    }
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_prompt_caching_async_stream_with_events_with_no_content(
    instrument_with_no_content,
    async_anthropic_client,
    span_exporter,
    log_exporter,
    reader,
):
    with open(Path(__file__).parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_anthropic_prompt_caching_async_stream <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        response = await async_anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_read_span = spans[1]

    assert cache_creation_span.attributes["gen_ai.usage.input_tokens"] == 1171
    assert cache_creation_span.attributes["gen_ai.usage.output_tokens"] == 290

    assert cache_read_span.attributes["gen_ai.usage.input_tokens"] == 1171
    assert cache_read_span.attributes["gen_ai.usage.output_tokens"] == 257

    # verify metrics
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-5-sonnet-20240620")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    i = 0
    for _ in range(2):
        # Validate the system message Event
        system_message_log = logs[i]
        assert_message_in_logs(system_message_log, "gen_ai.system.message", {})
        i += 1

        # Validate the user message Event
        user_message_log = logs[i]
        assert_message_in_logs(user_message_log, "gen_ai.user.message", {})
        i += 1

        # Validate the the ai response
        choice_event = {
            "index": 0,
            "finish_reason": "end_turn",
            "message": {},
        }
        assert_message_in_logs(logs[i], "gen_ai.choice", choice_event)
        i += 1


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
