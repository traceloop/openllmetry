from unittest.mock import MagicMock

import pytest
from ollama import Options
from opentelemetry.instrumentation.ollama.span_utils import (
    set_model_input_attributes,
    set_model_response_attributes,
)
from opentelemetry.sdk._logs import ReadableLogRecord
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
        options={
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 128,
            "seed": 7,
            "frequency_penalty": 0.25,
            "presence_penalty": 0.75,
            "stop": ["END", "STOP"],
        },
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0.5
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_P) == 0.9
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_K) == 40
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 128
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_SEED) == 7
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY) == 0.25
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY) == 0.75
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES) == ("END", "STOP")
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
        options=Options(temperature=0, seed=0, stop=["END"]),
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_SEED) == 0
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES) == ("END",)
    assert GenAIAttributes.GEN_AI_REQUEST_TOP_P not in ollama_span.attributes
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
        options={"temperature": 0.25, "num_predict": 64},
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0.25
    assert ollama_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 64
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


def _input_attributes(kwargs):
    span = MagicMock()
    span.is_recording.return_value = True
    set_model_input_attributes(span, kwargs)
    return {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}


def test_request_options_dict():
    attributes = _input_attributes(
        {
            "json": {
                "model": "llama3",
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 128,
                    "seed": 7,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.2,
                    "stop": ["END"],
                },
            }
        }
    )
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 128
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_SEED] == 7
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY] == 0.1
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY] == 0.2
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == ("END",)


def test_request_options_keep_zero_values():
    attributes = _input_attributes(
        {"json": {"options": {"temperature": 0, "top_p": 0.0, "num_predict": 0, "seed": 0}}}
    )
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.0
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 0
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_SEED] == 0


@pytest.mark.parametrize("options", [None, {}, "invalid", 5, ["temperature"]])
def test_request_options_absent_or_invalid_records_nothing(options):
    attributes = _input_attributes({"json": {"model": "llama3", "options": options}})
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "llama3"
    assert not [name for name in attributes if name.startswith("gen_ai.request.") and name != "gen_ai.request.model"]


def test_request_options_key_missing():
    attributes = _input_attributes({"json": {"model": "llama3"}})
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "llama3"
    assert GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE not in attributes


@pytest.mark.parametrize(
    "stop, expected",
    [
        ("END", ("END",)),
        (["A", "B"], ("A", "B")),
        (("A",), ("A",)),
        ("", None),
        ([], None),
        (None, None),
        (5, None),
        (["A", 5, "", None], ("A",)),
    ],
)
def test_request_options_stop(stop, expected):
    attributes = _input_attributes({"json": {"options": {"stop": stop}}})
    assert attributes.get(GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES) == expected


@pytest.mark.parametrize("num_predict", [-1, -2])
def test_request_options_unlimited_num_predict_is_not_recorded(num_predict):
    attributes = _input_attributes({"json": {"options": {"num_predict": num_predict}}})
    assert GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS not in attributes


def test_request_options_zero_num_predict_is_recorded():
    attributes = _input_attributes({"json": {"options": {"num_predict": 0}}})
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 0


def test_request_options_malformed_json_does_not_raise():
    span = MagicMock()
    span.is_recording.return_value = True
    set_model_input_attributes(span, {"json": {"model": "llama3", "options": {"temperature": object()}}})
    set_model_input_attributes(span, {"json": None})


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


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "ollama"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
