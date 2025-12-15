import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .utils import verify_metrics


@pytest.mark.vcr
def test_anthropic_thinking_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert (
        anthropic_span.attributes["gen_ai.completion.0.content"]
        == response.content[0].thinking
    )

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert (
        anthropic_span.attributes["gen_ai.completion.1.content"]
        == response.content[1].text
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_thinking_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }

    anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[user_message],
    )

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai thinking event
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": "Let me count the number of times the letter 'r' appears in the word \"strawberry\".\n\nThe "
            "word \"strawberry\" is spelled:\ns-t-r-a-w-b-e-r-r-y\n\nGoing through each letter:\ns - not an 'r'\nt - "
            "not an 'r'\nr - this is an 'r', so count = 1\na - not an 'r'\nw - not an 'r'\nb - not an 'r'\ne - not an "
            "'r'\nr - this is an 'r', so count = 2\nr - this is an 'r', so count = 3\ny - not an 'r'\n\nSo the letter "
            "'r' appears 3 times in the word \"strawberry\"."
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)

    # Validate the ai response event
    choice_event = {
        "index": 1,
        "finish_reason": "end_turn",
        "message": {
            "content": "The letter 'r' appears 3 times in the word \"strawberry\".",
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_thinking_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai thinking event
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)

    # Validate the ai response event
    choice_event = {
        "index": 1,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = await async_anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert (
        anthropic_span.attributes["gen_ai.completion.0.content"]
        == response.content[0].thinking
    )

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert (
        anthropic_span.attributes["gen_ai.completion.1.content"]
        == response.content[1].text
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }

    await async_anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[user_message],
    )

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai thinking event
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": "Let me count the number of times the letter 'r' appears in the word \"strawberry\".\n\nThe "
            "word \"strawberry\" is spelled: s-t-r-a-w-b-e-r-r-y\n\nLet me check for each 'r':\n1. The third letter is "
            "'r'\n2. The eighth letter is 'r'\n3. The ninth letter is 'r'\n\nSo there are 3 instances of the letter "
            "'r' in the word \"strawberry\"."
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)

    # Validate the ai response event
    choice_event = {
        "index": 1,
        "finish_reason": "end_turn",
        "message": {
            "content": "The letter 'r' appears 3 times in the word \"strawberry\".",
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_with_events_with_no_content(
    instrument_with_no_content,
    async_anthropic_client,
    span_exporter,
    log_exporter,
    reader,
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    await async_anthropic_client.messages.create(
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

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai thinking event
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)

    # Validate the ai response event
    choice_event = {
        "index": 1,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_thinking_streaming_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = anthropic_client.messages.create(
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
        elif (
            event.type == "content_block_delta" and event.delta.type == "thinking_delta"
        ):
            thinking += event.delta.thinking

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert anthropic_span.attributes["gen_ai.completion.1.content"] == text

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_thinking_streaming_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }

    response = anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        stream=True,
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[user_message],
    )

    text = ""
    thinking = ""

    for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            text += event.delta.text
        elif (
            event.type == "content_block_delta" and event.delta.type == "thinking_delta"
        ):
            thinking += event.delta.thinking

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai thinking event
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "thinking",
                "content": "Let me count the occurrences of the letter 'r' in the word \"strawberry\".\n\nThe word "
                "\"strawberry\" is spelled: s-t-r-a-w-b-e-r-r-y\n\nI'll go through each letter:\ns - not an 'r'\nt "
                "- not an 'r'\nr - this is an 'r', so count = 1\na - not an 'r'\nw - not an 'r'\nb - not an 'r'\ne - "
                "not an 'r'\nr - this is an 'r', so count = 2\nr - this is an 'r', so count = 3\ny - not an 'r'\n\nSo "
                "the letter 'r' appears 3 times in the word \"strawberry\".",
            }
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)

    # Validate the ai response event
    choice_event = {
        "index": 1,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "text",
                "content": "The letter 'r' appears 3 times in the word \"strawberry\".",
            }
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_thinking_streaming_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = anthropic_client.messages.create(
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
        elif (
            event.type == "content_block_delta" and event.delta.type == "thinking_delta"
        ):
            thinking += event.delta.thinking

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai thinking event
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)

    # Validate the ai response event
    choice_event = {
        "index": 1,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_streaming_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = await async_anthropic_client.messages.create(
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
        elif (
            event.type == "content_block_delta" and event.delta.type == "thinking_delta"
        ):
            thinking += event.delta.thinking

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert anthropic_span.attributes["gen_ai.completion.1.content"] == text

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_streaming_with_events_with_content(
    instrument_with_content, async_anthropic_client, span_exporter, log_exporter, reader
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }

    response = await async_anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        stream=True,
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[user_message],
    )

    text = ""
    thinking = ""

    async for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            text += event.delta.text
        elif (
            event.type == "content_block_delta" and event.delta.type == "thinking_delta"
        ):
            thinking += event.delta.thinking

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate user message Event
    user_message.pop("role", None)
    assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # Validate the ai thinking event
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "thinking",
                "content": "I need to count the occurrences of the letter 'r' in the word \"strawberry\".\n\nThe word "
                "is: s-t-r-a-w-b-e-r-r-y\n\nGoing through each letter:\n- s: not an 'r'\n- t: not an 'r'\n- r: this is "
                "an 'r' (first occurrence)\n- a: not an 'r'\n- w: not an 'r'\n- b: not an 'r'\n- e: not an 'r'\n- r: "
                "this is an 'r' (second occurrence)\n- r: this is an 'r' (third occurrence)\n- y: not an 'r'\n\nSo the "
                "letter 'r' appears 3 times in the word \"strawberry\".",
            }
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)

    # Validate the ai response event
    choice_event = {
        "index": 1,
        "finish_reason": "end_turn",
        "message": {
            "content": {
                "type": "text",
                "content": "The letter 'r' appears 3 times in the word \"strawberry\".\n\nYou can see them in the "
                "spelling: st(r)awbe(r)(r)y",
            }
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_streaming_with_events_with_no_content(
    instrument_with_no_content,
    async_anthropic_client,
    span_exporter,
    log_exporter,
    reader,
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = await async_anthropic_client.messages.create(
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
        elif (
            event.type == "content_block_delta" and event.delta.type == "thinking_delta"
        ):
            thinking += event.delta.thinking

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    verify_metrics(resource_metrics, "claude-3-7-sonnet-20250219")

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai thinking event
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)

    # Validate the ai response event
    choice_event = {
        "index": 1,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


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
