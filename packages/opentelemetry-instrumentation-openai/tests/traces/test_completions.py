import os
from unittest.mock import patch

import httpx
import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from .utils import assert_request_contains_tracecontext, spy_decorator


@pytest.mark.vcr
def test_completion(instrument_legacy, span_exporter, log_exporter, openai_client):
    openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq42D1Socatcl1rCmgYZOFX7dFZw"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_completion_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq42D1Socatcl1rCmgYZOFX7dFZw"
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
        "finish_reason": "length",
        "message": {"content": response.choices[0].text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_completion_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq42D1Socatcl1rCmgYZOFX7dFZw"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "length", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion(
    instrument_legacy, span_exporter, log_exporter, async_openai_client
):
    await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq43c8U5ZZCQBX5lrSpsANwcd3OF"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq43c8U5ZZCQBX5lrSpsANwcd3OF"
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
        "finish_reason": "length",
        "message": {
            "content": response.choices[0].text,
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_openai_client
):
    await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq43c8U5ZZCQBX5lrSpsANwcd3OF"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_completion_langchain_style(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    openai_client.completions.create(
        model="davinci-002",
        prompt=["Tell me a joke about opentelemetry"],
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq43QD6R2WqfxXLpYsRvSAIn9LB9"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_completion_langchain_style_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.completions.create(
        model="davinci-002",
        prompt=["Tell me a joke about opentelemetry"],
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq43QD6R2WqfxXLpYsRvSAIn9LB9"
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
        "finish_reason": "length",
        "message": {
            "content": response.choices[0].text,
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_completion_langchain_style_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    openai_client.completions.create(
        model="davinci-002",
        prompt=["Tell me a joke about opentelemetry"],
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq43QD6R2WqfxXLpYsRvSAIn9LB9"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_completion_streaming(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    # set os env for token usage record in stream mode
    original_value = os.environ.get("TRACELOOP_STREAM_TOKEN_USAGE")
    os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = "true"

    try:
        response = openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
            stream=True,
        )

        for _ in response:
            pass

        spans = span_exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            "openai.completion",
        ]
        open_ai_span = spans[0]
        assert (
            open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
            == "Tell me a joke about opentelemetry"
        )
        assert open_ai_span.attributes.get(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
        )
        assert (
            open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
            == "https://api.openai.com/v1/"
        )

        # check token usage attributes for stream
        completion_tokens = open_ai_span.attributes.get(
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
        )
        prompt_tokens = open_ai_span.attributes.get(
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS
        )
        total_tokens = open_ai_span.attributes.get(
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS
        )
        assert completion_tokens and prompt_tokens and total_tokens
        assert completion_tokens + prompt_tokens == total_tokens
        assert (
            open_ai_span.attributes.get("gen_ai.response.id")
            == "cmpl-8wq44ev1DvyhsBfm1hNwxfv6Dltco"
        )

        logs = log_exporter.get_finished_logs()
        assert (
            len(logs) == 0
        ), "Assert that it doesn't emit logs when use_legacy_attributes is True"
    finally:
        # unset env
        if original_value is None:
            del os.environ["TRACELOOP_STREAM_TOKEN_USAGE"]
        else:
            os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = original_value


@pytest.mark.vcr
def test_completion_streaming_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    # set os env for token usage record in stream mode
    original_value = os.environ.get("TRACELOOP_STREAM_TOKEN_USAGE")
    os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = "true"

    try:
        response = openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
            stream=True,
        )

        for _ in response:
            pass

        spans = span_exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            "openai.completion",
        ]
        open_ai_span = spans[0]
        assert (
            open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
            == "https://api.openai.com/v1/"
        )

        # check token usage attributes for stream
        completion_tokens = open_ai_span.attributes.get(
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
        )
        prompt_tokens = open_ai_span.attributes.get(
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS
        )
        total_tokens = open_ai_span.attributes.get(
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS
        )
        assert completion_tokens and prompt_tokens and total_tokens
        assert completion_tokens + prompt_tokens == total_tokens
        assert (
            open_ai_span.attributes.get("gen_ai.response.id")
            == "cmpl-8wq44ev1DvyhsBfm1hNwxfv6Dltco"
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
            "finish_reason": "length",
            "message": {
                "content": "-common.\n\nI'm a python microservice that reads a JSON configuration file in order"
            },
        }
        assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)
    finally:
        # unset env
        if original_value is None:
            del os.environ["TRACELOOP_STREAM_TOKEN_USAGE"]
        else:
            os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = original_value


@pytest.mark.vcr
def test_completion_streaming_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    # set os env for token usage record in stream mode
    original_value = os.environ.get("TRACELOOP_STREAM_TOKEN_USAGE")
    os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = "true"

    try:
        response = openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
            stream=True,
        )

        for _ in response:
            pass

        spans = span_exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            "openai.completion",
        ]
        open_ai_span = spans[0]
        assert (
            open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
            == "https://api.openai.com/v1/"
        )

        # check token usage attributes for stream
        completion_tokens = open_ai_span.attributes.get(
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
        )
        prompt_tokens = open_ai_span.attributes.get(
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS
        )
        total_tokens = open_ai_span.attributes.get(
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS
        )
        assert completion_tokens and prompt_tokens and total_tokens
        assert completion_tokens + prompt_tokens == total_tokens
        assert (
            open_ai_span.attributes.get("gen_ai.response.id")
            == "cmpl-8wq44ev1DvyhsBfm1hNwxfv6Dltco"
        )

        logs = log_exporter.get_finished_logs()
        assert len(logs) == 2

        # Validate user message Event
        user_message_log = logs[0]
        assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

        # Validate the ai response
        choice_event = {"index": 0, "finish_reason": "length", "message": {}}
        assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)
    finally:
        # unset env
        if original_value is None:
            del os.environ["TRACELOOP_STREAM_TOKEN_USAGE"]
        else:
            os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = original_value


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_streaming(
    instrument_legacy, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq44uFYuGm6kNe44ntRwluggKZFY"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_streaming_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq44uFYuGm6kNe44ntRwluggKZFY"
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
        "finish_reason": "length",
        "message": {
            "content": " that isn’t about collecting logs\n\nJ) Some of these folks helped bring the",
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_streaming_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "cmpl-8wq44uFYuGm6kNe44ntRwluggKZFY"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "length", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_completion_context_propagation(
    instrument_legacy, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.completions.create(
            # model="davinci-002",
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)
    assert (
        openai_span.attributes.get("gen_ai.response.id")
        == "cmpl-2996bf68f7f142fa817bdd32af678df9"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_completion_context_propagation_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.completions.create(
            # model="davinci-002",
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)
    assert (
        openai_span.attributes.get("gen_ai.response.id")
        == "cmpl-2996bf68f7f142fa817bdd32af678df9"
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
        "finish_reason": "length",
        "message": {
            "content": "\n\nI want to share an interesting story about opentelemetry. I'd like",
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_completion_context_propagation_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.completions.create(
            # model="davinci-002",
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)
    assert (
        openai_span.attributes.get("gen_ai.response.id")
        == "cmpl-2996bf68f7f142fa817bdd32af678df9"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "length", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_context_propagation(
    instrument_legacy, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)
    assert (
        openai_span.attributes.get("gen_ai.response.id")
        == "cmpl-4acc6171f6c34008af07ca8490da3b95"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_context_propagation_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)
    assert (
        openai_span.attributes.get("gen_ai.response.id")
        == "cmpl-4acc6171f6c34008af07ca8490da3b95"
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
        "finish_reason": "length",
        "message": {
            "content": "\n\nThere was a meter in a company that wanted to see improvement in the efficiency",
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_context_propagation_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)
    assert (
        openai_span.attributes.get("gen_ai.response.id")
        == "cmpl-4acc6171f6c34008af07ca8490da3b95"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "length", "message": {}}
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
