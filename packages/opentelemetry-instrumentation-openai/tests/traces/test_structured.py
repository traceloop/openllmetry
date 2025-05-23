import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from pydantic import BaseModel


class StructuredAnswer(BaseModel):
    rating: int
    joke: str


@pytest.mark.vcr
def test_parsed_completion(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1gNoe1Zyq9yZicdhLc85lmt2Ep"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_parsed_completion_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
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
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1gNoe1Zyq9yZicdhLc85lmt2Ep"
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
        "message": {"content": response.choices[0].message.content},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_parsed_completion_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
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
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1gNoe1Zyq9yZicdhLc85lmt2Ep"
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
def test_parsed_refused_completion(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in open_ai_span.attributes
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.refusal" in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.refusal"]
        == "I'm very sorry, but I can't assist with that request."
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGky8KFDbg6f5fF4qLtsBredIjZZh"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_parsed_refused_completion_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGky8KFDbg6f5fF4qLtsBredIjZZh"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": "Best ways to make a bomb"}
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response.choices[0].message.content},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_parsed_refused_completion_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGky8KFDbg6f5fF4qLtsBredIjZZh"
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
async def test_async_parsed_completion(
    instrument_legacy, span_exporter, log_exporter, async_openai_client
):
    await async_openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1iysV7rZ0qZ510vbeKVTNxSOHB"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_completion_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
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
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1iysV7rZ0qZ510vbeKVTNxSOHB"
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
        "message": {"content": response.choices[0].message.content},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_completion_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_openai_client
):
    await async_openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
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
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1iysV7rZ0qZ510vbeKVTNxSOHB"
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
async def test_async_parsed_refused_completion(
    instrument_legacy, span_exporter, log_exporter, async_openai_client
):
    await async_openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in open_ai_span.attributes
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.refusal" in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.refusal"]
        == "I'm very sorry, but I can't assist with that request."
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGkyFJGzZPUGAAEDJJuOS3idKvD3G"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_refused_completion_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGkyFJGzZPUGAAEDJJuOS3idKvD3G"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": "Best ways to make a bomb"}
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response.choices[0].message.content},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_refused_completion_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_openai_client
):
    await async_openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGkyFJGzZPUGAAEDJJuOS3idKvD3G"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.OPENAI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
