import asyncio
from unittest.mock import patch

import httpx
import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from .utils import assert_request_contains_tracecontext, spy_decorator


@pytest.mark.vcr
def test_chat(instrument_legacy, span_exporter, log_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get(
            SpanAttributes.LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
        )
        == "fp_2b778c6b35"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-908MD9ivBBLb6EaIjlqwFokntayQK"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get(
            SpanAttributes.LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
        )
        == "fp_2b778c6b35"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-908MD9ivBBLb6EaIjlqwFokntayQK"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": response.choices[0].message.content,
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get(
            SpanAttributes.LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
        )
        == "fp_2b778c6b35"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-908MD9ivBBLb6EaIjlqwFokntayQK"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_tool_calls(instrument_legacy, span_exporter, log_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.arguments"
        ]
        == '{"location": "San Francisco"}'
    )

    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "1"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9gKNZbUWSC4s2Uh2QfVV7PYiqWIuH"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_tool_calls_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes

    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9gKNZbUWSC4s2Uh2QfVV7PYiqWIuH"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate assistant message Event
    assistant_event = {
        "content": None,
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": '{"location": "San Francisco"}',
                },
            }
        ],
    }
    assert_message_in_logs(logs[0], "gen_ai.assistant.message", assistant_event)

    # Validate the tool message Event
    tool_event = {
        "content": "The weather in San Francisco is 70 degrees and sunny.",
    }
    assert_message_in_logs(logs[1], "gen_ai.tool.message", tool_event)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": "The weather in San Francisco is 70 degrees and sunny.",
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_tool_calls_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9gKNZbUWSC4s2Uh2QfVV7PYiqWIuH"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate assistant message Event
    assistant_event = {
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "get_current_weather"},
            }
        ]
    }
    assert_message_in_logs(logs[0], "gen_ai.assistant.message", assistant_event)

    # Validate the tool message Event
    tool_event = {}
    assert_message_in_logs(logs[1], "gen_ai.tool.message", tool_event)

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_pydantic_based_tool_calls(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    ChatCompletionMessageToolCall(
                        id="1",
                        type="function",
                        function=Function(
                            name="get_current_weather",
                            arguments='{"location": "San Francisco"}',
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.arguments"
        ]
        == '{"location": "San Francisco"}'
    )

    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "1"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9lvGJKrBUPeJjHi3KKSEbGfcfomOP"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_pydantic_based_tool_calls_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    ChatCompletionMessageToolCall(
                        id="1",
                        type="function",
                        function=Function(
                            name="get_current_weather",
                            arguments='{"location": "San Francisco"}',
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9lvGJKrBUPeJjHi3KKSEbGfcfomOP"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate assistant message Event
    assistant_event = {
        "content": None,
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": '{"location": "San Francisco"}',
                },
            }
        ],
    }
    assert_message_in_logs(logs[0], "gen_ai.assistant.message", assistant_event)

    # Validate the tool message Event
    tool_event = {
        "content": "The weather in San Francisco is 70 degrees and sunny.",
    }
    assert_message_in_logs(logs[1], "gen_ai.tool.message", tool_event)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": "The current weather in San Francisco is 70 degrees and sunny.",
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_pydantic_based_tool_calls_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    ChatCompletionMessageToolCall(
                        id="1",
                        type="function",
                        function=Function(
                            name="get_current_weather",
                            arguments='{"location": "San Francisco"}',
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9lvGJKrBUPeJjHi3KKSEbGfcfomOP"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate assistant message Event
    assistant_event = {
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "get_current_weather"},
            }
        ]
    }
    assert_message_in_logs(logs[0], "gen_ai.assistant.message", assistant_event)

    # Validate the tool message Event
    tool_event = {}
    assert_message_in_logs(logs[1], "gen_ai.tool.message", tool_event)

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_streaming(instrument_legacy, span_exporter, log_exporter, openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-908MECg5dMyTTbJEltubwQXeeWlBA"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_streaming_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-908MECg5dMyTTbJEltubwQXeeWlBA"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": (
                "Why did the opentelemetry developer go broke? \n"
                "Because they kept trying to trace their steps back too far!"
            ),
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_streaming_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-908MECg5dMyTTbJEltubwQXeeWlBA"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming(
    instrument_legacy, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    async for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9AGW3t9akkLW9f5f93B7mOhiqhNMC"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    async for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9AGW3t9akkLW9f5f93B7mOhiqhNMC"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": "Why did the developer break up with Opentelemetry? "
            "Because it couldn't handle the baggage of all their tracing requests!",
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    async for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    assert completion_tokens + prompt_tokens == total_tokens
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9AGW3t9akkLW9f5f93B7mOhiqhNMC"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
def test_with_asyncio_run(
    instrument_legacy, span_exporter, log_exporter, async_openai_client
):
    asyncio.run(
        async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-ANnyEsyt6uxfIIA7lcPLH95lKcEeK"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
def test_with_asyncio_run_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_openai_client
):
    asyncio.run(
        async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-ANnyEsyt6uxfIIA7lcPLH95lKcEeK"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": (
                "Why did the developer bring a compass to the Opentelemetry party? \n"
                "To ensure they didn't lose track of all their traces!"
            ),
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
def test_with_asyncio_run_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_openai_client
):
    asyncio.run(
        async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-ANnyEsyt6uxfIIA7lcPLH95lKcEeK"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_context_propagation(
    instrument_legacy, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chat-43f4347c3299481e9704ab77439fbdb8"
    )
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_context_propagation_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chat-43f4347c3299481e9704ab77439fbdb8"
    )
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": (
                "Why did the OpenTelemetry metric go to therapy?\n\n"
                'Because it was feeling a little "trapped" in its logging function, and wanted to "release" its stress.'
            ),
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_context_propagation_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chat-43f4347c3299481e9704ab77439fbdb8"
    )
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_context_propagation(
    instrument_legacy, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chat-4db07f02ecae49cbafe1d359db1650df"
    )
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_context_propagation_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chat-4db07f02ecae49cbafe1d359db1650df"
    )
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": (
                "A data scientist walks into an openTelemetry adoption meeting and says, \n\n"
                "\"I'm here to help track our progress, but I'm just a trace.\""
            ),
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_context_propagation_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chat-4db07f02ecae49cbafe1d359db1650df"
    )
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.OPENAI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


@pytest.mark.vcr
def test_chat_history_message_dict(span_exporter, openai_client):
    first_user_message = {
        "role": "user",
        "content": "Generate a random noun in Korean. Respond with just that word.",
    }
    second_user_message = {
        "role": "user",
        "content": "Now, generate a sentence using the word you just gave me.",
    }
    first_response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[first_user_message],
    )

    second_response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            first_user_message,
            {
                "role": "assistant",
                "content": first_response.choices[0].message.content,
            },
            second_user_message,
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 2
    first_span = spans[0]
    assert first_span.name == "openai.chat"
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == first_user_message["content"]
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == first_user_message["role"]
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == first_response.choices[0].message.content
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    )

    second_span = spans[1]
    assert second_span.name == "openai.chat"
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == first_user_message["content"]
    )
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == second_response.choices[0].message.content
    )
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == first_response.choices[0].message.content
    )
    assert second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]
        == second_user_message["content"]
    )
    assert second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "user"


@pytest.mark.vcr
def test_chat_history_message_pydantic(span_exporter, openai_client):
    first_user_message = {
        "role": "user",
        "content": "Generate a random noun in Korean. Respond with just that word.",
    }
    second_user_message = {
        "role": "user",
        "content": "Now, generate a sentence using the word you just gave me.",
    }
    first_response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[first_user_message],
    )

    second_response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            first_user_message,
            first_response.choices[0].message,
            second_user_message,
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 2
    first_span = spans[0]
    assert first_span.name == "openai.chat"
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == first_user_message["content"]
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == first_user_message["role"]
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == first_response.choices[0].message.content
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    )

    second_span = spans[1]
    assert second_span.name == "openai.chat"
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == first_user_message["content"]
    )
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == second_response.choices[0].message.content
    )
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == first_response.choices[0].message.content
    )
    assert second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]
        == second_user_message["content"]
    )
    assert second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "user"
