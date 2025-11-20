import pytest
from mistralai.models import UserMessage
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_mistralai_chat_legacy(
    instrument_legacy, mistralai_client, span_exporter, log_exporter
):
    response = mistralai_client.chat.complete(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response.choices[0].message.content
    )
    assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "e9459fcd56c742e0875167c9926c6aae"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_mistralai_chat_with_events_with_content(
    instrument_with_content, mistralai_client, span_exporter, log_exporter
):
    mistralai_client.chat.complete(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "864dc78a89b648cba3962647b8df4d3e"
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
            "content": (
                "Why did OpenTelemetry join a band?\n\n"
                "Because it wanted to hit those performance metrics right on the beat!\n\n"
                "(Bonus: It also has a wide range of instrumentation options, "
                "making it a versatile addition to any musical ensemble.)"
            ),
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_mistralai_chat_with_events_with_no_content(
    instrument_with_no_content, mistralai_client, span_exporter, log_exporter
):
    mistralai_client.chat.complete(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "b8b26a4542e14918b00f2886dc7913a6"
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
    gen = mistralai_client.chat.stream(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    for res in gen:
        response += res.data.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response
    )
    assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "6dc321029f5d4aa5899c1b38c9657a61"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_mistralai_streaming_chat_with_events_with_content(
    instrument_with_content, mistralai_client, span_exporter, log_exporter
):
    gen = mistralai_client.chat.stream(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    for res in gen:
        response += res.data.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "d3a0f557943648c49fb019bc65a64334"
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
            "content": (
                "Why did OpenTelemetry bring a map to the party?\n\n"
                "Because it wanted to trace all the connections!\n\n"
                "(This joke is for those who appreciate a dash of tech humor in their day.)"
            )
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_mistralai_streaming_chat_with_events_with_no_content(
    instrument_with_no_content, mistralai_client, span_exporter, log_exporter
):
    gen = mistralai_client.chat.stream(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    for res in gen:
        response += res.data.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "d663cffc326049acb5df51b0a1d60fb6"
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
    response = await mistralai_async_client.chat.complete_async(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response.choices[0].message.content
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "bda521177c084183bba5eaf32ad99027"
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
    await mistralai_async_client.chat.complete_async(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "cdd110acbcaf4cfc8a031783168d6389"
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
            "content": (
                "Here's a light-hearted joke about OpenTelemetry, a popular open-source system "
                "for generating, collecting, analyzing, and acting on telemetry data:\n\n"
                "Why did OpenTelemetry join a rock band?\n\n"
                "Because it wanted to monitor the metrics, span the genres, and distributed the beat!\n\n"
                "Of course, it's all in good fun and meant to be a playful way to explain the purpose "
                "of OpenTelemetry in a humorous manner. OpenTelemetry is a powerful tool for improving "
                "the performance, reliability, and efficiency of distributed systems, and it's essential "
                "for any modern software development."
            ),
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_chat_with_events_with_no_content(
    instrument_with_no_content, mistralai_async_client, span_exporter, log_exporter
):
    await mistralai_async_client.chat.complete_async(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-tiny"
    )
    # For some reason, async ollama chat doesn't report prompt token usage back
    # assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "d9f0293174c549028d43ab5f90607618"
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
    gen = await mistralai_async_client.chat.stream_async(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    async for res in gen:
        response += res.data.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert mistral_span.attributes.get("gen_ai.request.model") == "mistral-tiny"
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == response
    )
    assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "6dc321029f5d4aa5899c1b38c9657a61"
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
    gen = await mistralai_async_client.chat.stream_async(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    async for res in gen:
        response += res.data.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert mistral_span.attributes.get("gen_ai.request.model") == "mistral-tiny"
    assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "9c4b9456d7c642149a28f46bb36c3247"
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
            "content": (
                "Why did OpenTelemetry join a band?\n\n"
                "Because it wanted to help create beautiful, well-instrumented symphonies!\n\n"
                "(OpenTelemetry is an open-source, vendor-neutral observability solution for "
                "collecting, processing, and exporting telemetry data. It's often used in "
                "software development to monitor and improve application performance.)"
            )
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_streaming_chat_with_events_with_no_content(
    instrument_with_no_content, mistralai_async_client, span_exporter, log_exporter
):
    gen = await mistralai_async_client.chat.stream_async(
        model="mistral-tiny",
        messages=[
            UserMessage(content="Tell me a joke about OpenTelemetry"),
        ],
    )

    response = ""
    async for res in gen:
        response += res.data.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.chat"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert mistral_span.attributes.get("gen_ai.request.model") == "mistral-tiny"
    assert mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert mistral_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == mistral_span.attributes.get(
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
    ) + mistral_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "f5edd68bc31641f7a74d8d419da04b62"
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
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "mistral_ai"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
