import json
import pytest
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

GEN_AI_IS_STREAMING = SpanAttributes.GEN_AI_IS_STREAMING
GEN_AI_USAGE_TOTAL_TOKENS = SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS

MODEL = "llama3-8b-8192"
EXPECTED_SPAN_NAME = f"chat {MODEL}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_groq_span(spans):
    span = next((s for s in spans if s.name.startswith("chat ")), None)
    assert span is not None, f"No 'chat <model>' span found. Got: {[s.name for s in spans]}"
    return span


def _assert_otel_v2_span_attributes(span):
    """Assert the three core OTel 1.40 attributes are present on every span."""
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAIAttributes.GenAiProviderNameValues.GROQ.value
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == MODEL


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == GenAIAttributes.GenAiProviderNameValues.GROQ.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


# ---------------------------------------------------------------------------
# Legacy mode (use_legacy_attributes=True)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_chat_legacy(instrument_legacy, groq_client, span_exporter, log_exporter):
    groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [EXPECTED_SPAN_NAME]
    groq_span = _get_groq_span(spans)

    _assert_otel_v2_span_attributes(groq_span)

    input_messages = json.loads(groq_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["type"] == "text"
    assert input_messages[0]["parts"][0]["content"] == "Tell me a joke about opentelemetry"

    output_messages = json.loads(groq_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["finish_reason"] == "stop"
    assert output_messages[0]["parts"][0]["type"] == "text"
    assert output_messages[0]["parts"][0]["content"]

    assert groq_span.attributes.get(GEN_AI_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(GEN_AI_USAGE_TOTAL_TOKENS) > 0
    assert groq_span.attributes.get("gen_ai.response.id") == "chatcmpl-645691ff-34af-4d0f-a1c1-fe888f8685cc"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_with_events_with_content(instrument_with_content, groq_client, span_exporter, log_exporter):
    groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [EXPECTED_SPAN_NAME]
    groq_span = _get_groq_span(spans)

    _assert_otel_v2_span_attributes(groq_span)

    assert groq_span.attributes.get(GEN_AI_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(GEN_AI_USAGE_TOTAL_TOKENS) > 0
    assert groq_span.attributes.get("gen_ai.response.id") == "chatcmpl-645691ff-34af-4d0f-a1c1-fe888f8685cc"

    finish_reasons = groq_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
    assert finish_reasons is not None
    assert "stop" in finish_reasons

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

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
def test_chat_with_events_with_no_content(instrument_with_no_content, groq_client, span_exporter, log_exporter):
    groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [EXPECTED_SPAN_NAME]
    groq_span = _get_groq_span(spans)

    _assert_otel_v2_span_attributes(groq_span)

    assert groq_span.attributes.get(GEN_AI_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(GEN_AI_USAGE_TOTAL_TOKENS) > 0
    assert groq_span.attributes.get("gen_ai.response.id") == "chatcmpl-645691ff-34af-4d0f-a1c1-fe888f8685cc"

    finish_reasons = groq_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
    assert finish_reasons is not None
    assert "stop" in finish_reasons

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_legacy(instrument_legacy, async_groq_client, span_exporter, log_exporter):
    await async_groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [EXPECTED_SPAN_NAME]
    groq_span = _get_groq_span(spans)

    _assert_otel_v2_span_attributes(groq_span)

    input_messages = json.loads(groq_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["type"] == "text"
    assert input_messages[0]["parts"][0]["content"] == "Tell me a joke about opentelemetry"

    output_messages = json.loads(groq_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["finish_reason"] == "stop"
    assert output_messages[0]["parts"][0]["type"] == "text"
    assert output_messages[0]["parts"][0]["content"]

    assert groq_span.attributes.get(GEN_AI_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(GEN_AI_USAGE_TOTAL_TOKENS) > 0
    assert groq_span.attributes.get("gen_ai.response.id") == "chatcmpl-ec0a74e9-df7f-4e91-aa09-e9618451f5c9"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_with_events_with_content(
    instrument_with_content, async_groq_client, span_exporter, log_exporter
):
    await async_groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [EXPECTED_SPAN_NAME]
    groq_span = _get_groq_span(spans)

    _assert_otel_v2_span_attributes(groq_span)

    assert groq_span.attributes.get(GEN_AI_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(GEN_AI_USAGE_TOTAL_TOKENS) > 0
    assert groq_span.attributes.get("gen_ai.response.id") == "chatcmpl-ec0a74e9-df7f-4e91-aa09-e9618451f5c9"

    finish_reasons = groq_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
    assert finish_reasons is not None
    assert "stop" in finish_reasons

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

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
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [EXPECTED_SPAN_NAME]
    groq_span = _get_groq_span(spans)

    _assert_otel_v2_span_attributes(groq_span)

    assert groq_span.attributes.get(GEN_AI_IS_STREAMING) is False
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) > 0
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) > 0
    assert groq_span.attributes.get(GEN_AI_USAGE_TOTAL_TOKENS) > 0
    assert groq_span.attributes.get("gen_ai.response.id") == "chatcmpl-ec0a74e9-df7f-4e91-aa09-e9618451f5c9"

    finish_reasons = groq_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
    assert finish_reasons is not None
    assert "stop" in finish_reasons

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_chat_streaming_legacy(instrument_legacy, groq_client, span_exporter, log_exporter):
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [EXPECTED_SPAN_NAME]
    groq_span = _get_groq_span(spans)

    _assert_otel_v2_span_attributes(groq_span)

    input_messages = json.loads(groq_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages[0]["parts"][0]["content"] == "Tell me a joke about opentelemetry"

    output_messages = json.loads(groq_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["finish_reason"] == "stop"
    assert output_messages[0]["parts"][0]["content"] == content

    assert groq_span.attributes.get(GEN_AI_IS_STREAMING) is True
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 73
    assert groq_span.attributes.get(GEN_AI_USAGE_TOTAL_TOKENS) == 91

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_streaming_with_events_with_content(instrument_with_content, groq_client, span_exporter, log_exporter):
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [EXPECTED_SPAN_NAME]
    groq_span = _get_groq_span(spans)

    _assert_otel_v2_span_attributes(groq_span)

    assert groq_span.attributes.get(GEN_AI_IS_STREAMING) is True
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 73
    assert groq_span.attributes.get(GEN_AI_USAGE_TOTAL_TOKENS) == 91

    finish_reasons = groq_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
    assert finish_reasons is not None
    assert "stop" in finish_reasons

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

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
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [EXPECTED_SPAN_NAME]
    groq_span = _get_groq_span(spans)

    _assert_otel_v2_span_attributes(groq_span)

    assert groq_span.attributes.get(GEN_AI_IS_STREAMING) is True
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
    assert groq_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 73
    assert groq_span.attributes.get(GEN_AI_USAGE_TOTAL_TOKENS) == 91

    finish_reasons = groq_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
    assert finish_reasons is not None
    assert "stop" in finish_reasons

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)
