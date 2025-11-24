import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import \
    gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_writer_completions_legacy(
    instrument_legacy, writer_client, span_exporter, log_exporter
):
    response = writer_client.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        stop=["."],
        stream=False,
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 150
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        ".",
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].text
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_writer_completions_with_events_with_content(
    instrument_with_content, writer_client, span_exporter, log_exporter
):
    response = writer_client.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        stop=["."],
        stream=False,
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 150
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        ".",
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.choices[0].text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_completions_with_events_with_no_content(
    instrument_with_no_content, writer_client, span_exporter, log_exporter
):
    writer_client.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        stop=["."],
        stream=False,
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 150
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        ".",
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_streaming_completions_legacy(
    instrument_legacy, writer_client, span_exporter, log_exporter
):
    gen = writer_client.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop=["I am"],
        stream=True,
    )

    response = ""
    for res in gen:
        if res.value:
            response += res.value

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
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
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        "I am",
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
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
def test_writer_streaming_completions_with_events_with_content(
    instrument_with_content, writer_client, span_exporter, log_exporter
):
    gen = writer_client.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop=["I am"],
        stream=True,
    )

    response = ""
    for res in gen:
        if res.value:
            response += res.value

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
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
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        "I am",
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_writer_streaming_completions_with_events_with_no_content(
    instrument_with_no_content, writer_client, span_exporter, log_exporter
):
    gen = writer_client.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop=["I am"],
        stream=True,
    )

    response = ""
    for res in gen:
        if res.value:
            response += res.value

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
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
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        "I am",
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_completions_legacy(
    instrument_legacy, writer_client_async, span_exporter, log_exporter
):
    response = await writer_client_async.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        stop=["."],
        stream=False,
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]

    assert writer_span.name == "writerai.completions"
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 150
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        ".",
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].text
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_completions_with_events_with_content(
    instrument_with_content, writer_client_async, span_exporter, log_exporter
):
    response = await writer_client_async.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        stop=["."],
        stream=False,
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 150
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        ".",
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.choices[0].text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_completions_with_events_with_no_content(
    instrument_with_no_content, writer_client_async, span_exporter, log_exporter
):
    await writer_client_async.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        stop=["."],
        stream=False,
    )

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
    assert not writer_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}")
        == "palmyra-x4"
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}") == 150
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}") == 0.7
    )
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TOP_P}") == 0.9
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        ".",
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "writer"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_streaming_completions_legacy(
    instrument_legacy, writer_client_async, span_exporter, log_exporter
):
    gen = await writer_client_async.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop=["I am"],
        stream=True,
    )

    response = ""
    async for res in gen:
        if res.value:
            response += res.value

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
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
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        "I am",
    )

    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about OpenTelemetry"
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
async def test_writer_async_streaming_completions_with_events_with_content(
    instrument_with_content, writer_client_async, span_exporter, log_exporter
):
    gen = await writer_client_async.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop=["I am"],
        stream=True,
    )

    response = ""
    async for res in gen:
        if res.value:
            response += res.value

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
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
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        "I am",
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_streaming_completions_with_events_with_no_content(
    instrument_with_no_content, writer_client_async, span_exporter, log_exporter
):
    gen = await writer_client_async.completions.create(
        model="palmyra-x4",
        prompt="Tell me a joke about OpenTelemetry",
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop=["I am"],
        stream=True,
    )

    response = ""
    async for res in gen:
        if res.value:
            response += res.value

    spans = span_exporter.get_finished_spans()
    writer_span = spans[0]
    assert writer_span.name == "writerai.completions"
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "writer"
    assert (
        writer_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "completion"
    )
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
    assert writer_span.attributes.get(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}") == (
        "I am",
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)
