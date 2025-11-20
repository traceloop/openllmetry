import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.fixture
def openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
    ]


@pytest.mark.vcr
def test_open_ai_function_calls(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "What's the weather like in Boston?"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description"]
        == "Get the current weather in a given location"
    )
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-8wq4AUDD36geK9Za8cccowhObkV9H"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_open_ai_function_calls_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-8wq4AUDD36geK9Za8cccowhObkV9H"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "What's the weather like in Boston?"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "function_call",
        "message": {"content": None},
        "tool_calls": [
            {
                "id": "",
                "function": {
                    "arguments": '{\n  "location": "Boston"\n}',
                    "name": "get_current_weather",
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_open_ai_function_calls_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-8wq4AUDD36geK9Za8cccowhObkV9H"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "function_call",
        "message": {},
        "tool_calls": [
            {
                "id": "",
                "function": {"name": "get_current_weather"},
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_open_ai_function_calls_tools(
    instrument_legacy, span_exporter, log_exporter, openai_client, openai_tools
):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        tools=openai_tools,
        tool_choice="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "What's the weather like in Boston?"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description"]
        == "Get the current weather"
    )
    assert isinstance(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.id"],
        str,
    )
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-934OqhoorTmk1VnovIRXQCPk8PUTd"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_open_ai_function_calls_tools_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client, openai_tools
):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        tools=openai_tools,
        tool_choice="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-934OqhoorTmk1VnovIRXQCPk8PUTd"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "What's the weather like in Boston?"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {"content": None},
        "tool_calls": [
            {
                "id": "call_b4lU3wKOqwO7Xb2Ksok2XWyB",
                "function": {
                    "arguments": '{\n  "location": "Boston"\n}',
                    "name": "get_current_weather",
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_open_ai_function_calls_tools_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client, openai_tools
):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        tools=openai_tools,
        tool_choice="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-934OqhoorTmk1VnovIRXQCPk8PUTd"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {},
        "tool_calls": [
            {
                "id": "call_b4lU3wKOqwO7Xb2Ksok2XWyB",
                "function": {"name": "get_current_weather"},
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming(
    instrument_legacy, span_exporter, log_exporter, async_openai_client, openai_tools
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert isinstance(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.id"],
        str,
    )
    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason")
        == "tool_calls"
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.arguments"
        )
        == '{"location":"San Francisco, CA"}'
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4TmLd49mPoD6c0EnGlhNAp8b0on"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming_with_events_with_content(
    instrument_with_content,
    span_exporter,
    log_exporter,
    async_openai_client,
    openai_tools,
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4TmLd49mPoD6c0EnGlhNAp8b0on"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "What's the weather like in San Francisco?"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {"content": ""},
        "tool_calls": [
            {
                "function": {
                    "arguments": '{"location":"San Francisco, CA"}',
                    "name": "get_current_weather",
                },
                "id": "call_90R0NgAY0rQUYihSqVx7OuIs",
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming_with_events_with_no_content(
    instrument_with_no_content,
    span_exporter,
    log_exporter,
    async_openai_client,
    openai_tools,
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4TmLd49mPoD6c0EnGlhNAp8b0on"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {},
        "tool_calls": [
            {
                "id": "call_90R0NgAY0rQUYihSqVx7OuIs",
                "function": {"name": "get_current_weather"},
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_open_ai_function_calls_tools_parallel(
    instrument_legacy, span_exporter, log_exporter, openai_client, openai_tools
):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason")
        == "tool_calls"
    )

    assert isinstance(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.id"],
        str,
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.arguments"
        )
        == '{"location": "San Francisco"}'
    )

    assert isinstance(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.1.id"],
        str,
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.1.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.1.arguments"
        )
        == '{"location": "Boston"}'
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4cZhrW9CsqihSvXslk0EUtjASsO"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_open_ai_function_calls_tools_parallel_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client, openai_tools
):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4cZhrW9CsqihSvXslk0EUtjASsO"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "What's the weather like in San Francisco and Boston?"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {"content": None},
        "tool_calls": [
            {
                "function": {
                    "arguments": '{"location": "San Francisco"}',
                    "name": "get_current_weather",
                },
                "id": "call_3JNWJ9wdfRsmkhKWql4HqJhR",
                "type": "function",
            },
            {
                "function": {
                    "arguments": '{"location": "Boston"}',
                    "name": "get_current_weather",
                },
                "id": "call_8jQ7TzSBlLV4tzrMRpq5Tg98",
                "type": "function",
            },
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_open_ai_function_calls_tools_parallel_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client, openai_tools
):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4cZhrW9CsqihSvXslk0EUtjASsO"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {},
        "tool_calls": [
            {
                "function": {"name": "get_current_weather"},
                "id": "call_3JNWJ9wdfRsmkhKWql4HqJhR",
                "type": "function",
            },
            {
                "function": {"name": "get_current_weather"},
                "id": "call_8jQ7TzSBlLV4tzrMRpq5Tg98",
                "type": "function",
            },
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming_parallel(
    instrument_legacy, span_exporter, log_exporter, async_openai_client, openai_tools
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason")
        == "tool_calls"
    )

    assert isinstance(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.id"],
        str,
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.arguments"
        )
        == '{"location": "San Francisco"}'
    )

    assert isinstance(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.1.id"],
        str,
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.1.name"
        )
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes.get(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.1.arguments"
        )
        == '{"location": "Boston"}'
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g58noIjRkOeNNxfFsFfcNjhXlul7"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming_parallel_with_events_with_content(
    instrument_with_content,
    span_exporter,
    log_exporter,
    async_openai_client,
    openai_tools,
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g58noIjRkOeNNxfFsFfcNjhXlul7"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "What's the weather like in San Francisco and Boston?"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {"content": ""},
        "tool_calls": [
            {
                "function": {
                    "arguments": '{"location": "San Francisco"}',
                    "name": "get_current_weather",
                },
                "id": "call_cCPjAyfwTzboEKjVlqFrArNF",
                "type": "function",
            },
            {
                "function": {
                    "arguments": '{"location": "Boston"}',
                    "name": "get_current_weather",
                },
                "id": "call_Zi4He1Ns0mozwT6f85nW1BOW",
                "type": "function",
            },
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming_parallel_with_events_with_no_content(
    instrument_with_no_content,
    span_exporter,
    log_exporter,
    async_openai_client,
    openai_tools,
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g58noIjRkOeNNxfFsFfcNjhXlul7"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {},
        "tool_calls": [
            {
                "function": {"name": "get_current_weather"},
                "id": "call_cCPjAyfwTzboEKjVlqFrArNF",
                "type": "function",
            },
            {
                "function": {"name": "get_current_weather"},
                "id": "call_Zi4He1Ns0mozwT6f85nW1BOW",
                "type": "function",
            },
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


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
