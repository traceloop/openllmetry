from unittest.mock import MagicMock

import pytest
from opentelemetry.instrumentation.ollama.span_utils import (
    set_model_response_attributes,
)
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes


@pytest.mark.vcr
def test_ollama_chat_legacy(
    instrument_legacy, ollama_client, span_exporter, log_exporter
):
    response = ollama_client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response["message"]["content"]
    )
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_ollama_chat_with_events_with_content(
    instrument_with_content, ollama_client, span_exporter, log_exporter
):
    response = ollama_client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response["message"]["content"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_ollama_chat_with_events_with_no_content(
    instrument_with_no_content, ollama_client, span_exporter, log_exporter
):
    ollama_client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_ollama_chat_tool_calls_legacy(
    instrument_legacy, ollama_client, span_exporter, log_exporter
):
    ollama_client.chat(
        model="llama3.1",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {"location": "San Francisco"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3.1"
    )
    assert (
        f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.content"
        not in ollama_span.attributes
    )
    assert (
        ollama_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        ollama_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.tool_calls.0.arguments"]
        == '{"location": "San Francisco"}'
    )

    assert (
        ollama_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_ollama_chat_tool_calls_with_events_with_content(
    instrument_with_content, ollama_client, span_exporter, log_exporter
):
    response = ollama_client.chat(
        model="llama3.1",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {"location": "San Francisco"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]

    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3.1"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate assistant message Event
    user_message_log = logs[0]
    assistant_message = {
        "content": {},
        "tool_calls": [
            {
                "id": "",
                "function": {
                    "arguments": {"location": "San Francisco"},
                    "name": "get_current_weather",
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(
        user_message_log, "gen_ai.assistant.message", assistant_message
    )

    # Validate the tool message Event
    assert_message_in_logs(
        logs[1],
        "gen_ai.tool.message",
        {"content": "The weather in San Francisco is 70 degrees and sunny."},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response["message"]["content"]},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_ollama_chat_tool_calls_with_events_with_no_content(
    instrument_with_no_content, ollama_client, span_exporter, log_exporter
):
    ollama_client.chat(
        model="llama3.1",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {"location": "San Francisco"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]

    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3.1"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate assistant message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.assistant.message",
        {
            "tool_calls": [
                {
                    "id": "",
                    "function": {"name": "get_current_weather"},
                    "type": "function",
                }
            ]
        },
    )

    # Validate the tool message Event
    assert_message_in_logs(logs[1], "gen_ai.tool.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_ollama_streaming_chat_legacy(
    instrument_legacy, ollama_client, span_exporter, log_exporter
):
    gen = ollama_client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        stream=True,
    )

    response = ""
    for res in gen:
        response += res["message"]["content"]

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response
    )
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_ollama_streaming_chat_with_events_with_content(
    instrument_with_content, ollama_client, span_exporter, log_exporter
):
    gen = ollama_client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        stream=True,
    )

    response = ""
    for res in gen:
        response += res["message"]["content"]

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_ollama_streaming_chat_with_events_with_no_content(
    instrument_with_no_content, ollama_client, span_exporter, log_exporter
):
    gen = ollama_client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        stream=True,
    )

    response = ""
    for res in gen:
        response += res["message"]["content"]

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_chat_legacy(
    instrument_legacy, ollama_client_async, span_exporter, log_exporter
):
    response = await ollama_client_async.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response["message"]["content"]
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_chat_with_events_with_content(
    instrument_with_content, ollama_client_async, span_exporter, log_exporter
):
    response = await ollama_client_async.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response["message"]["content"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_chat_with_events_with_no_content(
    instrument_with_no_content, ollama_client_async, span_exporter, log_exporter
):
    await ollama_client_async.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_streaming_chat_legacy(
    instrument_legacy, ollama_client_async, span_exporter, log_exporter
):
    gen = await ollama_client_async.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        stream=True,
    )

    response = ""
    async for res in gen:
        response += res["message"]["content"]

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response
    )
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_streaming_chat_with_events_with_content(
    instrument_with_content, ollama_client_async, span_exporter, log_exporter
):
    gen = await ollama_client_async.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        stream=True,
    )

    response = ""
    async for res in gen:
        response += res["message"]["content"]

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_streaming_chat_with_events_with_no_content(
    instrument_with_no_content, ollama_client_async, span_exporter, log_exporter
):
    gen = await ollama_client_async.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        stream=True,
    )

    response = ""
    async for res in gen:
        response += res["message"]["content"]

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + ollama_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_token_histogram_recording():
    span = MagicMock()
    token_histogram = MagicMock()
    llm_request_type = LLMRequestTypeValues.COMPLETION
    response = {
        "model": "llama3",
        "prompt_eval_count": 7,
        "eval_count": 10,
    }
    set_model_response_attributes(span, token_histogram, llm_request_type, response)
    token_histogram.record.assert_any_call(
        7,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "Ollama",
            GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
            GenAIAttributes.GEN_AI_RESPONSE_MODEL: "llama3",
        },
    )
    token_histogram.record.assert_any_call(
        10,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "Ollama",
            GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
            GenAIAttributes.GEN_AI_RESPONSE_MODEL: "llama3",
        },
    )


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "ollama"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
