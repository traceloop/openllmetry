from unittest.mock import patch

import httpx
import openai
import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode

from .utils import assert_request_contains_tracecontext, spy_decorator


@pytest.mark.vcr
def test_embeddings(instrument_legacy, span_exporter, log_exporter, openai_client):
    openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-embedding-ada-002"
    )
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_embeddings_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-embedding-ada-002"
    )
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
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
        "finish_reason": "unknown",
        "message": {"content": response.data[0].embedding},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_embeddings_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-embedding-ada-002"
    )
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_embeddings_with_raw_response(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    response = openai_client.embeddings.with_raw_response.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )

    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-embedding-ada-002"
    )
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )

    parsed_response = response.parse()
    assert parsed_response.data[0]

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_embeddings_with_raw_response_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.embeddings.with_raw_response.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-embedding-ada-002"
    )
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )

    parsed_response = response.parse()
    assert parsed_response.data[0]

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
        "finish_reason": "unknown",
        "message": {"content": parsed_response.data[0].embedding},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_embeddings_with_raw_response_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    response = openai_client.embeddings.with_raw_response.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-embedding-ada-002"
    )
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )

    parsed_response = response.parse()
    assert parsed_response.data[0]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_azure_openai_embeddings(instrument_legacy, span_exporter, log_exporter):
    api_key = "test-api-key"
    azure_resource = "test-resource"
    azure_deployment = "test-deployment"

    openai_client = openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=f"https://{azure_resource}.openai.azure.com",
        azure_deployment=azure_deployment,
        api_version="2023-07-01-preview",
    )
    openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="embedding",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "embedding"
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == f"https://{azure_resource}.openai.azure.com/openai/deployments/{azure_deployment}/"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_VERSION]
        == "2023-07-01-preview"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_azure_openai_embeddings_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    api_key = "test-api-key"
    azure_resource = "test-resource"
    azure_deployment = "test-deployment"

    openai_client = openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=f"https://{azure_resource}.openai.azure.com",
        azure_deployment=azure_deployment,
        api_version="2023-07-01-preview",
    )
    response = openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="embedding",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "embedding"
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == f"https://{azure_resource}.openai.azure.com/openai/deployments/{azure_deployment}/"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_VERSION]
        == "2023-07-01-preview"
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
        "finish_reason": "unknown",
        "message": {"content": response.data[0].embedding},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_azure_openai_embeddings_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    api_key = "test-api-key"
    azure_resource = "test-resource"
    azure_deployment = "test-deployment"

    openai_client = openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=f"https://{azure_resource}.openai.azure.com",
        azure_deployment=azure_deployment,
        api_version="2023-07-01-preview",
    )
    openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="embedding",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "embedding"
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == f"https://{azure_resource}.openai.azure.com/openai/deployments/{azure_deployment}/"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_VERSION]
        == "2023-07-01-preview"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_embeddings_context_propagation(
    instrument_legacy, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_embeddings_context_propagation_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        response = vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

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
        "finish_reason": "unknown",
        "message": {"content": response.data[0].embedding},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_embeddings_context_propagation_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_embeddings_context_propagation(
    instrument_legacy, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_embeddings_context_propagation_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        response = await async_vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

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
        "finish_reason": "unknown",
        "message": {"content": response.data[0].embedding},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_embeddings_context_propagation_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def test_embeddings_exception(instrument_legacy, span_exporter, openai_client):
    openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="text-embedding-ada-002",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert open_ai_span.status.description.startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")


@pytest.mark.asyncio
async def test_async_embeddings_exception(instrument_legacy, span_exporter, async_openai_client):
    async_openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        await async_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="text-embedding-ada-002",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert open_ai_span.status.description.startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")


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
