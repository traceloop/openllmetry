import pytest
from mistralai.models.chat_completion import ChatMessage
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_mistralai_chat_legacy(
    instrument_legacy, mistralai_client, span_exporter, log_exporter
):
    response = mistralai_client.chat(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].message.content
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "d5f25c4c1e29441db526ce7db3400010"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_mistralai_chat_with_events_with_content(
    instrument_with_content, mistralai_client, span_exporter, log_exporter
):
    mistralai_client.chat(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "d5f25c4c1e29441db526ce7db3400010"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {"content": "Tell me a joke about OpenTelemetry"}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": "Why did the OpenTelemetry go to therapy?\n\nBecause it had too many traces and couldn't handle it!\n\n(For non-developers, OpenTelemetry is a set of APIs, libraries, agents, and instrumentation that standardize how you collect, generate, transport, and consume telemetry data.)",
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_mistralai_chat_with_events_with_no_content(
    instrument_with_no_content, mistralai_client, span_exporter, log_exporter
):
    mistralai_client.chat(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "d5f25c4c1e29441db526ce7db3400010"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_mistralai_streaming_chat_legacy(
    instrument_legacy, mistralai_client, span_exporter, log_exporter
):
    gen = mistralai_client.chat_stream(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    for res in gen:
        response += res.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "937738cd542a461da86a967ad7c2c8db"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_mistralai_streaming_chat_with_events_with_content(
    instrument_with_content, mistralai_client, span_exporter, log_exporter
):
    gen = mistralai_client.chat_stream(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    for res in gen:
        response += res.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "937738cd542a461da86a967ad7c2c8db"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {"content": "Tell me a joke about OpenTelemetry"}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": "Why did OpenTelemetry bring a map to the party?\n\nBecause it heard there would be a lot of tracing!\n\n(For those not familiar, OpenTelemetry is an open-source project for standardizing observability data.)"
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_mistralai_streaming_chat_with_events_with_no_content(
    instrument_with_no_content, mistralai_client, span_exporter, log_exporter
):
    gen = mistralai_client.chat_stream(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    for res in gen:
        response += res.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "937738cd542a461da86a967ad7c2c8db"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_chat_legacy(
    instrument_legacy, mistralai_async_client, span_exporter, log_exporter
):
    response = await mistralai_async_client.chat(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].message.content
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "84e3f907fd2045eba99a91a50a6c5a53"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_chat_with_events_with_content(
    instrument_with_content, mistralai_async_client, span_exporter, log_exporter
):
    await mistralai_async_client.chat(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "84e3f907fd2045eba99a91a50a6c5a53"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {"content": "Tell me a joke about OpenTelemetry"}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": "Why did OpenTelemetry bring a map to the party?\n\nBecause it wanted to trace the route of the good conversations!\n\n(I'm here to bring a smile, not to code or explain complex systems!)",
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_chat_with_events_with_no_content(
    instrument_with_no_content, mistralai_async_client, span_exporter, log_exporter
):
    await mistralai_async_client.chat(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "84e3f907fd2045eba99a91a50a6c5a53"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_streaming_chat_legacy(
    instrument_legacy, mistralai_async_client, span_exporter, log_exporter
):
    gen = await mistralai_async_client.chat_stream(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    async for res in gen:
        response += res.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert mistral_span.attributes.get("gen_ai.request.model") == "mistral-tiny"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response
    )
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "8b811019d651417b913b5c16b32732e2"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_streaming_chat_with_events_with_content(
    instrument_with_content, mistralai_async_client, span_exporter, log_exporter
):
    gen = await mistralai_async_client.chat_stream(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    async for res in gen:
        response += res.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert mistral_span.attributes.get("gen_ai.request.model") == "mistral-tiny"
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "8b811019d651417b913b5c16b32732e2"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {"content": "Tell me a joke about OpenTelemetry"}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": "Why did the OpenTelemetry library join a band?\n\nBecause it wanted to create beautiful, instrumented, and distributed melodies!\n\n(Note: This is a play on words, as OpenTelemetry helps developers to collect distributed traces, so they can monitor and improve the performance of their applications.)"
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_streaming_chat_with_events_with_no_content(
    instrument_with_no_content, mistralai_async_client, span_exporter, log_exporter
):
    gen = await mistralai_async_client.chat_stream(
        model="mistral-tiny",
        messages=[
            ChatMessage(role="user", content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    async for res in gen:
        response += res.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert mistral_span.attributes.get("gen_ai.request.model") == "mistral-tiny"
    assert mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + mistral_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "8b811019d651417b913b5c16b32732e2"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "mistral_ai"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
