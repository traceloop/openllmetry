import time
import uuid

import httpx
import openai
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import Meters
from opentelemetry.trace import StatusCode
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


def _respond(request: httpx.Request) -> httpx.Response:
    # Shape copied from a recorded Responses API reply.
    return httpx.Response(200, json={
        "id": f"resp_{uuid.uuid4().hex}", "object": "response", "created_at": 0, "status": "completed",
        "model": "gpt-4.1-nano-2025-04-14",
        "output": [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": '{"name":"Alice","age":30}', "annotations": []}]}],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    })


def _fail(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _background(*statuses):
    """Serve one background response that reports each status in turn, failing on a None status."""
    response_id = f"resp_{uuid.uuid4().hex}"
    replies = iter(statuses)

    def respond(request: httpx.Request) -> httpx.Response:
        status = next(replies)
        if status is None:
            return _fail(request)
        body = _respond(request).json() | {"id": response_id, "status": status}
        if status != "completed":
            # A queued, in-progress or cancelled background response has no output or usage yet.
            body |= {"output": [], "usage": None}
        return httpx.Response(200, json=body)

    return response_id, respond


@pytest.fixture
def metric_points(reader):
    reader.get_metrics_data()  # drop points left by earlier tests

    def collect():
        points = {}
        data = reader.get_metrics_data()
        for rm in data.resource_metrics if data else []:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    points.setdefault(metric.name, []).extend(metric.data.data_points)
        return points

    return collect


def _client(handler):
    return openai.OpenAI(
        api_key="test-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def _async_client(handler):
    return openai.AsyncOpenAI(
        api_key="test-key", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def _assert_starts_with_span(duration, span, finished_ns):
    # Recorded after the span ended and before the call returned, so the metric must
    # measure from the span's start time.
    assert (span.end_time - span.start_time) / 1e9 <= duration.sum <= (finished_ns - span.start_time) / 1e9


def _assert_success_metrics(points, span_exporter, finished_ns):
    [span] = span_exporter.get_finished_spans()
    [duration] = points[Meters.LLM_OPERATION_DURATION]
    assert duration.count == 1
    _assert_starts_with_span(duration, span, finished_ns)
    assert ERROR_TYPE not in duration.attributes
    assert duration.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert duration.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4.1-nano-2025-04-14"
    assert duration.attributes["server.address"] == "https://api.openai.com/v1/"
    assert duration.attributes["stream"] is False

    tokens = {
        p.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]: p.sum for p in points.get(Meters.LLM_TOKEN_USAGE, [])
    }
    assert tokens == {"input": 7, "output": 3}


def _assert_error_metrics(points, span_exporter, stream, finished_ns):
    [span] = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == "APIConnectionError"

    [duration] = points[Meters.LLM_OPERATION_DURATION]
    assert duration.count == 1
    _assert_starts_with_span(duration, span, finished_ns)
    assert duration.attributes[ERROR_TYPE] == "APIConnectionError"
    # No response exists, so the request model is reported.
    assert duration.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4.1-nano"
    assert duration.attributes["stream"] is stream
    assert Meters.LLM_TOKEN_USAGE not in points


def test_responses_create_records_duration_and_tokens(instrument_legacy, span_exporter, metric_points):
    _client(_respond).responses.create(model="gpt-4.1-nano", input="hello")

    _assert_success_metrics(metric_points(), span_exporter, time.time_ns())


@pytest.mark.asyncio
async def test_async_responses_create_records_duration_and_tokens(instrument_legacy, span_exporter, metric_points):
    await _async_client(_respond).responses.create(model="gpt-4.1-nano", input="hello")

    _assert_success_metrics(metric_points(), span_exporter, time.time_ns())


def test_responses_parse_records_duration_and_tokens(instrument_legacy, span_exporter, metric_points):
    _client(_respond).responses.parse(model="gpt-4.1-nano", input="Alice is 30", text_format=Person)

    _assert_success_metrics(metric_points(), span_exporter, time.time_ns())


@pytest.mark.asyncio
async def test_async_responses_parse_records_duration_and_tokens(instrument_legacy, span_exporter, metric_points):
    await _async_client(_respond).responses.parse(model="gpt-4.1-nano", input="Alice is 30", text_format=Person)

    _assert_success_metrics(metric_points(), span_exporter, time.time_ns())


@pytest.mark.parametrize("stream", [False, True])
def test_responses_create_error_records_duration_with_error_type(
    instrument_legacy, span_exporter, metric_points, stream
):
    with pytest.raises(openai.APIConnectionError):
        _client(_fail).responses.create(model="gpt-4.1-nano", input="hello", stream=stream)

    _assert_error_metrics(metric_points(), span_exporter, stream, time.time_ns())


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_async_responses_create_error_records_duration_with_error_type(
    instrument_legacy, span_exporter, metric_points, stream
):
    with pytest.raises(openai.APIConnectionError):
        await _async_client(_fail).responses.create(model="gpt-4.1-nano", input="hello", stream=stream)

    _assert_error_metrics(metric_points(), span_exporter, stream, time.time_ns())


_retrieve_calls = pytest.mark.parametrize(
    "retrieve",
    [
        lambda responses, response_id: responses.retrieve(response_id),
        lambda responses, response_id: responses.retrieve(response_id=response_id),
    ],
    ids=["positional", "keyword"],
)


def test_reread_of_finished_response_records_no_duration(instrument_legacy, metric_points):
    response_id, respond = _background("queued", "completed", "completed", "completed")
    client = _client(respond)
    client.responses.create(model="gpt-4.1-nano", input="hello", background=True)
    client.responses.retrieve(response_id)
    metric_points()  # drop the operation's own point

    client.responses.retrieve(response_id)
    client.responses.cancel(response_id)
    assert not metric_points().get(Meters.LLM_OPERATION_DURATION)


@pytest.mark.asyncio
async def test_async_reread_of_finished_response_records_no_duration(instrument_legacy, metric_points):
    response_id, respond = _background("queued", "completed", "completed", "completed")
    client = _async_client(respond)
    await client.responses.create(model="gpt-4.1-nano", input="hello", background=True)
    await client.responses.retrieve(response_id)
    metric_points()  # drop the operation's own point

    await client.responses.retrieve(response_id)
    await client.responses.cancel(response_id)
    assert not metric_points().get(Meters.LLM_OPERATION_DURATION)


def _assert_untracked_spans_only(points, span_exporter):
    # The terminal retrieve and the cancel still end their spans, as before.
    assert len(span_exporter.get_finished_spans()) == 2
    assert not points.get(Meters.LLM_OPERATION_DURATION)
    assert not points.get(Meters.LLM_TOKEN_USAGE)


def test_untracked_retrieve_records_no_duration(instrument_legacy, span_exporter, metric_points):
    # Responses another process created: polled here to completion, polled then cancelled, and only cancelled.
    polled_id, polled = _background("queued", "completed")
    _client(polled).responses.retrieve(polled_id)
    _client(polled).responses.retrieve(polled_id)
    cancelled_id, cancelled = _background("queued", "cancelled")
    _client(cancelled).responses.retrieve(cancelled_id)
    _client(cancelled).responses.cancel(cancelled_id)
    untracked_id, untracked = _background("cancelled")
    _client(untracked).responses.cancel(untracked_id)

    _assert_untracked_spans_only(metric_points(), span_exporter)


@pytest.mark.asyncio
async def test_async_untracked_retrieve_records_no_duration(instrument_legacy, span_exporter, metric_points):
    polled_id, polled = _background("queued", "completed")
    await _async_client(polled).responses.retrieve(polled_id)
    await _async_client(polled).responses.retrieve(polled_id)
    cancelled_id, cancelled = _background("queued", "cancelled")
    await _async_client(cancelled).responses.retrieve(cancelled_id)
    await _async_client(cancelled).responses.cancel(cancelled_id)
    untracked_id, untracked = _background("cancelled")
    await _async_client(untracked).responses.cancel(untracked_id)

    _assert_untracked_spans_only(metric_points(), span_exporter)


def _assert_failed_poll_span_only(points, span_exporter):
    [span] = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == "APIConnectionError"
    assert not points.get(Meters.LLM_OPERATION_DURATION)
    span_exporter.clear()


@_retrieve_calls
def test_failed_poll_records_no_duration(instrument_legacy, span_exporter, metric_points, retrieve):
    response_id, respond = _background("queued", None, "completed")
    client = _client(respond)
    client.responses.create(model="gpt-4.1-nano", input="hello", background=True)
    with pytest.raises(openai.APIConnectionError):
        retrieve(client.responses, response_id)

    _assert_failed_poll_span_only(metric_points(), span_exporter)
    retrieve(client.responses, response_id)
    _assert_success_metrics(metric_points(), span_exporter, time.time_ns())


@pytest.mark.asyncio
@_retrieve_calls
async def test_async_failed_poll_records_no_duration(instrument_legacy, span_exporter, metric_points, retrieve):
    response_id, respond = _background("queued", None, "completed")
    client = _async_client(respond)
    await client.responses.create(model="gpt-4.1-nano", input="hello", background=True)
    with pytest.raises(openai.APIConnectionError):
        await retrieve(client.responses, response_id)

    _assert_failed_poll_span_only(metric_points(), span_exporter)
    await retrieve(client.responses, response_id)
    _assert_success_metrics(metric_points(), span_exporter, time.time_ns())
