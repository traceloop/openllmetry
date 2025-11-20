import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_chat_legacy(instrument_legacy, groq_client, span_exporter, log_exporter):
    groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert (
        groq_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert groq_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
    assert (
        groq_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-645691ff-34af-4d0f-a1c1-fe888f8685cc"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_with_events_with_content(
    instrument_with_content, groq_client, span_exporter, log_exporter
):
    groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
    assert (
        groq_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-645691ff-34af-4d0f-a1c1-fe888f8685cc"
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
            "content": "Here's a joke about OpenTelemetry:\n\nWhy did OpenTelemetry go to the doctor?\n\nBecause it "
            'was feeling a little "lost in the traces"!\n\nI hope that brings a trace to your eye!'
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_with_events_with_no_content(
    instrument_with_no_content, groq_client, span_exporter, log_exporter
):
    groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
    assert (
        groq_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-645691ff-34af-4d0f-a1c1-fe888f8685cc"
    )

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
async def test_async_chat_legacy(
    instrument_legacy, async_groq_client, span_exporter, log_exporter
):
    await async_groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert (
        groq_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert groq_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
    assert (
        groq_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-ec0a74e9-df7f-4e91-aa09-e9618451f5c9"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_with_events_with_content(
    instrument_with_content, async_groq_client, span_exporter, log_exporter
):
    await async_groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
    assert (
        groq_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-ec0a74e9-df7f-4e91-aa09-e9618451f5c9"
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
            "content": "A joke about OpenTelemetry! Here's one:\n\nWhy did the OpenTelemetry agent go to therapy?\n\n"
            'Because it was struggling to "trace" its emotions and "span" its feelings of frustration!\n\n(Sorry, '
            'it\'s a bit of a " metrics"-al pun, but I hope it "counted" as a good joke!)'
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_with_events_with_no_content(
    instrument_with_no_content, async_groq_client, span_exporter, log_exporter
):
    await async_groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
    assert (
        groq_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-ec0a74e9-df7f-4e91-aa09-e9618451f5c9"
    )

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
def test_chat_streaming_legacy(
    instrument_legacy, groq_client, span_exporter, log_exporter
):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert (
        groq_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        groq_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == content
    )
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 73
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) == 91

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_streaming_with_events_with_content(
    instrument_with_content, groq_client, span_exporter, log_exporter
):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 73
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) == 91

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
            "content": "A joke about OpenTelemetry!\n\nWhy did the OpenTelemetry Span go to therapy?\n\nBecause it "
            'was feeling "traced" out and needed to "inject" some self-care into its life!\n\n(Sorry, it\'s a bit of '
            'a " instrumentation"-ally-challenged joke, but I hope it "exposes" a smile on your face!)'
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_streaming_with_events_with_no_content(
    instrument_with_no_content, groq_client, span_exporter, log_exporter
):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 73
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) == 91

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


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.GROQ.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
