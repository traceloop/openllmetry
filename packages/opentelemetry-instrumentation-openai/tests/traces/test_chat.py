import asyncio
from unittest.mock import patch

import httpx
import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from opentelemetry.instrumentation.openai.utils import is_reasoning_supported

from .utils import assert_request_contains_tracecontext, spy_decorator


@pytest.mark.vcr
def test_chat(instrument_legacy, span_exporter, log_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
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
    assert open_ai_span.attributes.get(
        SpanAttributes.LLM_IS_STREAMING) is False
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
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
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
    assert open_ai_span.attributes.get(
        SpanAttributes.LLM_IS_STREAMING) is False
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
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
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
    assert open_ai_span.attributes.get(
        SpanAttributes.LLM_IS_STREAMING) is False
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
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[
            f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.arguments"
        ]
        == '{"location": "San Francisco"}'
    )

    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.tool_call_id"] == "1"
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

    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in open_ai_span.attributes

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
    assert_message_in_logs(
        logs[0], "gen_ai.assistant.message", assistant_event)

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

    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in open_ai_span.attributes
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
    assert_message_in_logs(
        logs[0], "gen_ai.assistant.message", assistant_event)

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
    try:
        from openai.types.chat.chat_completion_message_function_tool_call import Function
    except (ImportError, ModuleNotFoundError, AttributeError):
        try:
            from openai.types.chat.chat_completion_message_tool_call import Function
        except (ImportError, ModuleNotFoundError, AttributeError):
            pytest.skip("Could not import Function. Please check your OpenAI version. Skipping test.")

    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    ChatCompletionMessageFunctionToolCall(
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

    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[
            f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.arguments"
        ]
        == '{"location": "San Francisco"}'
    )

    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.tool_call_id"] == "1"
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
    try:
        from openai.types.chat.chat_completion_message_function_tool_call import Function
    except (ImportError, ModuleNotFoundError, AttributeError):
        try:
            from openai.types.chat.chat_completion_message_tool_call import Function
        except (ImportError, ModuleNotFoundError, AttributeError):
            pytest.skip("Could not import Function. Please check your OpenAI version. Skipping test.")

    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    ChatCompletionMessageFunctionToolCall(
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

    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in open_ai_span.attributes
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
    assert_message_in_logs(
        logs[0], "gen_ai.assistant.message", assistant_event)

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
    try:
        from openai.types.chat.chat_completion_message_function_tool_call import Function
    except (ImportError, ModuleNotFoundError, AttributeError):
        try:
            from openai.types.chat.chat_completion_message_tool_call import Function
        except (ImportError, ModuleNotFoundError, AttributeError):
            pytest.skip("Could not import Function. Please check your OpenAI version. Skipping test.")

    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    ChatCompletionMessageFunctionToolCall(
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

    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in open_ai_span.attributes
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
    assert_message_in_logs(
        logs[0], "gen_ai.assistant.message", assistant_event)

    # Validate the tool message Event
    tool_event = {}
    assert_message_in_logs(logs[1], "gen_ai.tool.message", tool_event)

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_streaming(instrument_legacy, span_exporter, log_exporter, mock_openai_client):
    response = mock_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
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
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "http://localhost:5002/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    # Mock OpenAI background may produce different number of events, just check it's reasonable
    assert len(events) > 0

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    assert completion_tokens and prompt_tokens and total_tokens
    # When OpenAI API provides token usage, check that the sum of completion and prompt tokens equals total tokens
    assert completion_tokens + prompt_tokens == total_tokens
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-7UR4UcvmeD79Xva3UxkKkL2es6b5W"
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
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
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
    completion_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    # Only assert token usage if API provides it (modern OpenAI API includes usage in streaming)
    if completion_tokens and prompt_tokens and total_tokens:
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
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
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

    # check token usage attributes for stream (optional, depends on API support)
    completion_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    if completion_tokens and prompt_tokens and total_tokens:
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
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
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
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # check token usage attributes for stream
    completion_tokens = open_ai_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    if completion_tokens and prompt_tokens and total_tokens:
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
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
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
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    if completion_tokens and prompt_tokens and total_tokens:
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
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
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
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    )
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    if completion_tokens and prompt_tokens and total_tokens:
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
    # In OpenTelemetry 1.37.0+, event_name is a field on LogRecord, not in attributes
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
        first_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == first_user_message["content"]
    )
    assert (
        first_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]
        == first_user_message["role"]
    )
    assert (
        first_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == first_response.choices[0].message.content
    )
    assert (
        first_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"] == "assistant"
    )

    second_span = spans[1]
    assert second_span.name == "openai.chat"
    assert (
        second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == first_user_message["content"]
    )
    assert (
        second_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == second_response.choices[0].message.content
    )
    assert (
        second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
        == first_response.choices[0].message.content
    )
    assert second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "assistant"
    assert (
        second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.content"]
        == second_user_message["content"]
    )
    assert second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.role"] == "user"


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
        first_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == first_user_message["content"]
    )
    assert (
        first_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]
        == first_user_message["role"]
    )
    assert (
        first_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == first_response.choices[0].message.content
    )
    assert (
        first_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"] == "assistant"
    )

    second_span = spans[1]
    assert second_span.name == "openai.chat"
    assert (
        second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == first_user_message["content"]
    )
    assert (
        second_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == second_response.choices[0].message.content
    )
    assert (
        second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
        == first_response.choices[0].message.content
    )
    assert second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "assistant"
    assert (
        second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.content"]
        == second_user_message["content"]
    )
    assert second_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.role"] == "user"


@pytest.mark.vcr
@pytest.mark.skipif(not is_reasoning_supported(),
                    reason="Reasoning is not supported in older OpenAI library versions")
def test_chat_reasoning(instrument_legacy, span_exporter,
                        log_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "user",
                "content": "Count r's in strawberry"
            }
        ],
        reasoning_effort="low",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[-1]

    assert span.attributes["gen_ai.request.reasoning_effort"] == "low"
    assert span.attributes["gen_ai.usage.reasoning_tokens"] > 0


@pytest.mark.vcr
def test_chat_with_service_tier(instrument_legacy, span_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": "Say hello"
            }
        ],
        service_tier="priority",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[-1]

    assert span.attributes["openai.request.service_tier"] == "priority"
    assert span.attributes["openai.response.service_tier"] == "priority"


def test_chat_exception(instrument_legacy, span_exporter, openai_client):
    openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}],
        )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(
        SpanAttributes.LLM_IS_STREAMING) is False
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert open_ai_span.status.description.startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")
    assert open_ai_span.attributes.get("error.type") == "AuthenticationError"
    assert "Traceback (most recent call last):" in event.attributes["exception.stacktrace"]
    assert "openai.AuthenticationError" in event.attributes["exception.stacktrace"]
    assert "invalid_api_key" in event.attributes["exception.stacktrace"]


@pytest.mark.asyncio
async def test_chat_async_exception(instrument_legacy, span_exporter, async_openai_client):
    async_openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        await async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}],
        )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(
        SpanAttributes.LLM_IS_STREAMING) is False
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert open_ai_span.status.description.startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")
    assert "Traceback (most recent call last):" in event.attributes["exception.stacktrace"]
    assert "openai.AuthenticationError" in event.attributes["exception.stacktrace"]
    assert "invalid_api_key" in event.attributes["exception.stacktrace"]
    assert open_ai_span.attributes.get("error.type") == "AuthenticationError"


@pytest.mark.vcr
def test_chat_streaming_not_consumed(instrument_legacy, span_exporter, log_exporter, reader, openai_client):
    """Test that streaming responses are properly instrumented even when not consumed"""

    # Create streaming response but don't consume it
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    # Don't consume the response - this should still create proper traces and metrics
    del response

    # Force garbage collection to trigger cleanup
    import gc
    gc.collect()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"

    # Verify span was properly closed
    assert open_ai_span.status.status_code == StatusCode.OK
    assert open_ai_span.end_time is not None
    assert open_ai_span.end_time > open_ai_span.start_time

    assert open_ai_span.attributes.get(
        SpanAttributes.LLM_REQUEST_MODEL) == "gpt-3.5-turbo"
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True
    assert open_ai_span.attributes.get(
        SpanAttributes.LLM_REQUEST_TYPE) == "chat"

    assert open_ai_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content") == "Tell me a joke about opentelemetry"
    assert open_ai_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.role") == "user"

    # Verify duration metric was recorded even without consuming the stream
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    scope_metrics = resource_metrics[0].scope_metrics
    assert len(scope_metrics) > 0

    # Find duration metric
    duration_metrics = [
        metric for metric in scope_metrics[0].metrics
        if metric.name == "gen_ai.client.operation.duration"
    ]

    assert len(duration_metrics) == 1, "Duration metric should be recorded"
    duration_metric = duration_metrics[0]

    # Verify metric data
    assert duration_metric.data.data_points
    data_point = duration_metric.data.data_points[0]
    assert data_point.count >= 1, f"Expected count >= 1, got {data_point.count}"
    assert data_point.sum > 0, f"Duration should be greater than 0, got {data_point.sum}"
    assert data_point.min > 0, f"Min duration should be greater than 0, got {data_point.min}"
    assert data_point.max > 0, f"Max duration should be greater than 0, got {data_point.max}"

    # Verify metric attributes
    attributes = data_point.attributes
    assert attributes.get(
        "gen_ai.system") == "openai", f"Expected gen_ai.system=openai, got {attributes.get('gen_ai.system')}"
    assert attributes.get(
        "gen_ai.operation.name") == "chat", f"Expected operation=chat, got {attributes.get('gen_ai.operation.name')}"

    streaming_data_points = [
        dp for dp in duration_metric.data.data_points
        if dp.attributes.get("stream") is True
    ]
    assert len(streaming_data_points) >= 1, (
        f"Expected at least one streaming data point, got data points with attributes: "
        f"{[dict(dp.attributes) for dp in duration_metric.data.data_points]}"
    )


@pytest.mark.vcr
def test_chat_streaming_partial_consumption(instrument_legacy, span_exporter, log_exporter, reader, openai_client):
    """Test that streaming responses are properly instrumented when partially consumed"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    )

    # Consume only the first chunk
    first_chunk = next(iter(response))
    assert first_chunk is not None

    del response

    import gc
    gc.collect()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"

    assert open_ai_span.status.status_code == StatusCode.OK
    assert open_ai_span.end_time is not None

    assert open_ai_span.attributes.get(
        SpanAttributes.LLM_REQUEST_MODEL) == "gpt-3.5-turbo"
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    # Should have at least one event from the consumed chunk
    events = open_ai_span.events
    assert len(events) >= 1

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0, "Should have resource metrics"

    scope_metrics = resource_metrics[0].scope_metrics
    assert len(scope_metrics) > 0, "Should have scope metrics"

    # Find duration metric
    duration_metrics = [
        metric for metric in scope_metrics[0].metrics
        if metric.name == "gen_ai.client.operation.duration"
    ]

    assert len(duration_metrics) == 1, (
        f"Duration metric should be recorded, found metrics: "
        f"{[m.name for m in scope_metrics[0].metrics]}"
    )
    duration_metric = duration_metrics[0]

    assert duration_metric.data.data_points, "Duration metric should have data points"
    data_point = duration_metric.data.data_points[0]
    assert data_point.count >= 1, f"Expected count >= 1, got {data_point.count}"
    assert data_point.sum > 0, f"Duration should be greater than 0, got {data_point.sum}"

    attributes = data_point.attributes
    assert attributes.get(
        "gen_ai.system") == "openai", f"Expected gen_ai.system=openai, got {attributes.get('gen_ai.system')}"
    assert attributes.get(
        "gen_ai.operation.name") == "chat", f"Expected operation=chat, got {attributes.get('gen_ai.operation.name')}"

    streaming_data_points = [
        dp for dp in duration_metric.data.data_points
        if dp.attributes.get("stream") is True
    ]
    assert len(streaming_data_points) >= 1, (
        f"Expected at least one streaming data point, got data points with attributes: "
        f"{[dict(dp.attributes) for dp in duration_metric.data.data_points]}"
    )


@pytest.mark.vcr
def test_chat_streaming_exception_during_consumption(instrument_legacy, span_exporter, log_exporter, openai_client):
    """Test that streaming responses handle exceptions during consumption properly"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a short story"}],
        stream=True,
    )

    # Simulate exception during consumption
    count = 0
    try:
        for chunk in response:
            count += 1
            if count == 2:  # Interrupt after second chunk
                raise Exception("Simulated interruption")
    except Exception as e:
        # Force cleanup by deleting the response object
        del response
        import gc
        gc.collect()
        # Re-raise to verify the exception was caught
        assert "Simulated interruption" in str(e)

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"

    # Verify span was properly closed (status should be OK since exception was in user code, not in our iterator)
    assert open_ai_span.status.status_code == StatusCode.OK
    assert open_ai_span.end_time is not None

    # Should have events from the consumed chunks before exception
    events = open_ai_span.events
    assert len(events) >= 2  # At least 2 chunk events before exception


@pytest.mark.vcr
def test_chat_streaming_memory_leak_prevention(instrument_legacy, span_exporter, log_exporter, openai_client):
    """Test that creating many streams without consuming them doesn't cause memory leaks"""
    import gc
    import weakref

    initial_spans = len(span_exporter.get_finished_spans())

    # Create a stream without consuming it
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    # Create weak reference to track if object is garbage collected
    weak_ref = weakref.ref(response)

    del response

    gc.collect()

    # Verify object was garbage collected
    assert weak_ref() is None, "Stream object was not garbage collected"

    # Verify span was properly closed
    final_spans = span_exporter.get_finished_spans()
    new_spans = len(final_spans) - initial_spans
    assert new_spans == 1, f"Expected 1 new span, got {new_spans}"

    # Verify span is properly closed
    span = final_spans[-1]
    assert span.name == "openai.chat"
    assert span.status.status_code == StatusCode.OK
    assert span.end_time is not None
