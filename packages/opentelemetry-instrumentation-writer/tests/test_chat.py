import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import \
    event_attributes as EventAttributes
from opentelemetry.semconv._incubating.attributes import \
    gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_writer_chat_legacy(
    instrument_legacy, writer_client, span_exporter, log_exporter
):
    response = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].message.content
    )
    assert writer_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 40
    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_writer_chat_with_events_with_content(
    instrument_with_content, writer_client, span_exporter, log_exporter
):
    response = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response.choices[0].message.content, "tool_calls": []},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_chat_with_events_with_no_content(
    instrument_with_no_content, writer_client, span_exporter, log_exporter
):
    writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"tool_calls": []},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_chat_tool_calls_legacy(
    instrument_legacy, writer_client, span_exporter, log_exporter
):
    response = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "What is the weather like today?"
    )
    assert (
        f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.content"
        not in writer_span.attributes
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.arguments"]
        == '{"location": "San Francisco"}'
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].message.content
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_writer_chat_tool_calls_with_events_with_content(
    instrument_with_content, writer_client, span_exporter, log_exporter
):
    response = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "What is the weather like today?"},
    )

    assistant_message = {
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "arguments": '{"location": "San Francisco"}',
                    "name": "get_current_weather",
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.assistant.message", assistant_message)

    assert_message_in_logs(
        logs[2],
        "gen_ai.tool.message",
        {"content": "The weather in San Francisco is 70 degrees and sunny."},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response.choices[0].message.content},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_chat_tool_calls_with_events_with_no_content(
    instrument_with_no_content, writer_client, span_exporter, log_exporter
):
    writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    assert_message_in_logs(
        logs[1],
        "gen_ai.assistant.message",
        {
            "tool_calls": [
                {
                    "function": {"name": "get_current_weather"},
                    "type": "function",
                }
            ],
        },
    )

    assert_message_in_logs(logs[2], "gen_ai.tool.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_streaming_chat_legacy(
    instrument_legacy, writer_client, span_exporter, log_exporter
):
    gen = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=True,
        stream_options={"include_usage": True},
    )

    response = ""
    for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )
    assert writer_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 39
    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_writer_streaming_chat_with_events_with_content(
    instrument_with_content, writer_client, span_exporter, log_exporter
):
    gen = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=True,
        stream_options={"include_usage": True},
    )

    response = ""
    for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 39
    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response, "tool_calls": []},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_streaming_chat_with_events_with_no_content(
    instrument_with_no_content, writer_client, span_exporter, log_exporter
):
    gen = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=True,
        stream_options={"include_usage": True},
    )

    response = ""
    for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"tool_calls": []},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_streaming_chat_tool_calls_legacy(
    instrument_legacy, writer_client, span_exporter, log_exporter
):
    gen = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=True,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
        stream_options={"include_usage": True},
    )

    response = ""
    for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "What is the weather like today?"
    )
    assert (
        f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.content"
        not in writer_span.attributes
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.arguments"]
        == '{"location": "San Francisco"}'
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_writer_streaming_chat_tool_calls_with_events_with_content(
    instrument_with_content, writer_client, span_exporter, log_exporter
):
    gen = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=True,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
        stream_options={"include_usage": True},
    )

    response = ""
    for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "What is the weather like today?"},
    )

    assistant_message = {
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "arguments": '{"location": "San Francisco"}',
                    "name": "get_current_weather",
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.assistant.message", assistant_message)

    assert_message_in_logs(
        logs[2],
        "gen_ai.tool.message",
        {"content": "The weather in San Francisco is 70 degrees and sunny."},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response, "tool_calls": []},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_streaming_chat_tool_calls_with_events_with_no_content(
    instrument_with_no_content, writer_client, span_exporter, log_exporter
):
    gen = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=True,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
        stream_options={"include_usage": True},
    )

    response = ""
    for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    assert_message_in_logs(
        logs[1],
        "gen_ai.assistant.message",
        {
            "tool_calls": [
                {
                    "function": {"name": "get_current_weather"},
                    "type": "function",
                }
            ],
        },
    )

    assert_message_in_logs(logs[2], "gen_ai.tool.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"tool_calls": []},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_chat_tool_call_request_legacy(
    instrument_legacy, writer_client, span_exporter, log_exporter
):
    response = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like in Zakopane today?"},
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "What is the weather like in Zakopane today?"
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason"]
        == response.choices[0].finish_reason
    )
    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id"]
        == response.choices[0].message.tool_calls[0].id
    )
    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"]
        == response.choices[0].message.tool_calls[0].function.name
    )
    assert (
        writer_span.attributes[
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"
        ]
        == response.choices[0].message.tool_calls[0].function.arguments
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == response.choices[0].message.role
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_writer_chat_tool_call_request_with_events_with_content(
    instrument_with_content, writer_client, span_exporter, log_exporter
):
    response = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like in Zakopane today?"},
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "What is the weather like in Zakopane today?"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": response.choices[0].finish_reason,
        "message": {
            "content": response.choices[0].message.content,
            "tool_calls": [
                {
                    "id": response.choices[0].message.tool_calls[0].id,
                    "function": {
                        "arguments": response.choices[0]
                        .message.tool_calls[0]
                        .function.arguments,
                        "name": response.choices[0].message.tool_calls[0].function.name,
                    },
                    "type": response.choices[0].message.tool_calls[0].type,
                    "index": None,
                }
            ],
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_chat_tool_call_request_with_events_with_no_content(
    instrument_with_no_content, writer_client, span_exporter, log_exporter
):
    response = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like in Zakopane today?"},
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": response.choices[0].finish_reason,
        "message": {
            "tool_calls": [
                {
                    "id": response.choices[0].message.tool_calls[0].id,
                    "function": {
                        "name": response.choices[0].message.tool_calls[0].function.name,
                    },
                    "type": response.choices[0].message.tool_calls[0].type,
                    "index": None,
                }
            ],
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_chat_legacy(
    instrument_legacy, writer_client_async, span_exporter, log_exporter
):
    response = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].message.content
    )
    assert writer_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 40
    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_chat_with_events_with_content(
    instrument_with_content, writer_client_async, span_exporter, log_exporter
):
    response = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 40
    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response.choices[0].message.content, "tool_calls": []},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_chat_with_events_with_no_content(
    instrument_with_no_content, writer_client_async, span_exporter, log_exporter
):
    await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"tool_calls": []},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_chat_tool_calls_legacy(
    instrument_legacy, writer_client_async, span_exporter, log_exporter
):
    response = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "What is the weather like today?"
    )
    assert (
        f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.content"
        not in writer_span.attributes
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.arguments"]
        == '{"location": "San Francisco"}'
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].message.content
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_chat_tool_calls_with_events_with_content(
    instrument_with_content, writer_client_async, span_exporter, log_exporter
):
    response = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "What is the weather like today?"},
    )

    assistant_message = {
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "arguments": '{"location": "San Francisco"}',
                    "name": "get_current_weather",
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.assistant.message", assistant_message)

    assert_message_in_logs(
        logs[2],
        "gen_ai.tool.message",
        {"content": "The weather in San Francisco is 70 degrees and sunny."},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response.choices[0].message.content},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_chat_tool_calls_with_events_with_no_content(
    instrument_with_no_content, writer_client_async, span_exporter, log_exporter
):
    await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    assert_message_in_logs(
        logs[1],
        "gen_ai.assistant.message",
        {
            "tool_calls": [
                {
                    "function": {"name": "get_current_weather"},
                    "type": "function",
                }
            ],
        },
    )

    assert_message_in_logs(logs[2], "gen_ai.tool.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_streaming_chat_legacy(
    instrument_legacy, writer_client_async, span_exporter, log_exporter
):
    gen = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=True,
        stream_options={"include_usage": True},
    )

    response = ""
    async for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )
    assert writer_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 39
    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_streaming_chat_with_events_with_content(
    instrument_with_content, writer_client_async, span_exporter, log_exporter
):
    gen = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=True,
        stream_options={"include_usage": True},
    )

    response = ""
    async for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 39
    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response, "tool_calls": []},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_streaming_chat_with_events_with_no_content(
    instrument_with_no_content, writer_client_async, span_exporter, log_exporter
):
    gen = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=True,
        stream_options={"include_usage": True},
    )

    response = ""
    async for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"tool_calls": []},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "writer"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_streaming_chat_tool_calls_legacy(
    instrument_legacy, writer_client_async, span_exporter, log_exporter
):
    gen = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=True,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
        stream_options={"include_usage": True},
    )

    response = ""
    async for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "What is the weather like today?"
    )
    assert (
        f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.content"
        not in writer_span.attributes
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.arguments"]
        == '{"location": "San Francisco"}'
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_streaming_chat_tool_calls_with_events_with_content(
    instrument_with_content, writer_client_async, span_exporter, log_exporter
):
    gen = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=True,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
        stream_options={"include_usage": True},
    )

    response = ""
    async for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "What is the weather like today?"},
    )

    assistant_message = {
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "arguments": '{"location": "San Francisco"}',
                    "name": "get_current_weather",
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.assistant.message", assistant_message)

    assert_message_in_logs(
        logs[2],
        "gen_ai.tool.message",
        {"content": "The weather in San Francisco is 70 degrees and sunny."},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response, "tool_calls": []},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_streaming_chat_tool_calls_with_events_with_no_content(
    instrument_with_no_content, writer_client_async, span_exporter, log_exporter
):
    gen = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like today?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
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
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=True,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
        stream_options={"include_usage": True},
    )

    response = ""
    async for res in gen:
        if res.choices and res.choices[0].message.content:
            response += res.choices[0].message.content

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    assert_message_in_logs(
        logs[1],
        "gen_ai.assistant.message",
        {
            "tool_calls": [
                {
                    "function": {"name": "get_current_weather"},
                    "type": "function",
                }
            ],
        },
    )

    assert_message_in_logs(logs[2], "gen_ai.tool.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"tool_calls": []},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_chat_tool_call_request_legacy(
    instrument_legacy, writer_client_async, span_exporter, log_exporter
):
    response = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like in Zakopane today?"},
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "What is the weather like in Zakopane today?"
    )

    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason"]
        == response.choices[0].finish_reason
    )
    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id"]
        == response.choices[0].message.tool_calls[0].id
    )
    assert (
        writer_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"]
        == response.choices[0].message.tool_calls[0].function.name
    )
    assert (
        writer_span.attributes[
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"
        ]
        == response.choices[0].message.tool_calls[0].function.arguments
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == response.choices[0].message.role
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_chat_tool_call_request_with_events_with_content(
    instrument_with_content, writer_client_async, span_exporter, log_exporter
):
    response = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like in Zakopane today?"},
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "What is the weather like in Zakopane today?"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": response.choices[0].finish_reason,
        "message": {
            "content": response.choices[0].message.content,
            "tool_calls": [
                {
                    "id": response.choices[0].message.tool_calls[0].id,
                    "function": {
                        "arguments": response.choices[0]
                        .message.tool_calls[0]
                        .function.arguments,
                        "name": response.choices[0].message.tool_calls[0].function.name,
                    },
                    "type": response.choices[0].message.tool_calls[0].type,
                    "index": None,
                }
            ],
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_chat_tool_call_request_with_events_with_no_content(
    instrument_with_no_content, writer_client_async, span_exporter, log_exporter
):
    response = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {"role": "user", "content": "What is the weather like in Zakopane today?"},
        ],
        tools=[
            {
                "function": {
                    "description": "Return weather in the specific location.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "Location to return weather at",
                                "type": "string",
                            },
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        stream=False,
        max_tokens=340,
        temperature=0.7,
        top_p=0.9,
        stop="I am",
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.chat"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Writer"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 340
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}")
        == "I am"
    )

    assert writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + writer_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": response.choices[0].finish_reason,
        "message": {
            "tool_calls": [
                {
                    "id": response.choices[0].message.tool_calls[0].id,
                    "function": {
                        "name": response.choices[0].message.tool_calls[0].function.name,
                    },
                    "type": response.choices[0].message.tool_calls[0].type,
                    "index": None,
                }
            ],
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)
