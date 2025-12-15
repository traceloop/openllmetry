import json
import pytest
from pydantic import BaseModel
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import StatusCode


class StructuredAnswer(BaseModel):
    rating: int
    joke: str


@pytest.mark.vcr
def test_parsed_completion(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1gNoe1Zyq9yZicdhLc85lmt2Ep"
    )

    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_parsed_completion_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.chat.completions.parse(
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
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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
    openai_client.chat.completions.parse(
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
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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
    openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in open_ai_span.attributes
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.refusal" in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.refusal"]
        == "I'm very sorry, but I can't assist with that request."
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGky8KFDbg6f5fF4qLtsBredIjZZh"
    )
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_parsed_refused_completion_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGky8KFDbg6f5fF4qLtsBredIjZZh"
    )
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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
    openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGky8KFDbg6f5fF4qLtsBredIjZZh"
    )
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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
    await async_openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1iysV7rZ0qZ510vbeKVTNxSOHB"
    )
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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
    response = await async_openai_client.chat.completions.parse(
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
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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
    await async_openai_client.chat.completions.parse(
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
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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
    await async_openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in open_ai_span.attributes
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.refusal" in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.refusal"]
        == "I'm very sorry, but I can't assist with that request."
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGkyFJGzZPUGAAEDJJuOS3idKvD3G"
    )
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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
    response = await async_openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGkyFJGzZPUGAAEDJJuOS3idKvD3G"
    )
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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
    await async_openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGkyFJGzZPUGAAEDJJuOS3idKvD3G"
    )
    assert (
        json.loads(open_ai_span.attributes.get("gen_ai.request.structured_output_schema"))
        == StructuredAnswer.model_json_schema()
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


def test_parsed_completion_exception(
    instrument_legacy, span_exporter, openai_client
):
    openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        openai_client.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
            response_format=StructuredAnswer,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span: Span = spans[0]
    assert span.name == "openai.chat"
    assert span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE) == "https://api.openai.com/v1/"
    assert span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content") == "Tell me a joke about opentelemetry"
    assert span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.role") == "user"

    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description.startswith("Error code: 401")
    events = span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")
    assert "Traceback (most recent call last):" in event.attributes["exception.stacktrace"]
    assert "openai.AuthenticationError" in event.attributes["exception.stacktrace"]
    assert "invalid_api_key" in event.attributes["exception.stacktrace"]
    assert span.attributes.get("error.type") == "AuthenticationError"


@pytest.mark.asyncio
async def test_async_parsed_completion_exception(
    instrument_legacy, span_exporter, async_openai_client
):
    async_openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        await async_openai_client.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
            response_format=StructuredAnswer,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span: Span = spans[0]
    assert span.name == "openai.chat"
    assert span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE) == "https://api.openai.com/v1/"
    assert span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content") == "Tell me a joke about opentelemetry"
    assert span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.role") == "user"

    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description.startswith("Error code: 401")
    events = span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")
    assert "Traceback (most recent call last):" in event.attributes["exception.stacktrace"]
    assert "openai.AuthenticationError" in event.attributes["exception.stacktrace"]
    assert "invalid_api_key" in event.attributes["exception.stacktrace"]
    assert span.attributes.get("error.type") == "AuthenticationError"
