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


def _unparseable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "application/json"}, content=b"{not json")


_RAW_CALLS = [
    ("create", (), {"model": "gpt-4.1-nano", "input": "hello"}),
    ("parse", (), {"model": "gpt-4.1-nano", "input": "Alice is 30", "text_format": Person}),
    ("retrieve", ("resp_1",), {}),
    ("cancel", ("resp_1",), {}),
]


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
        p.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]: p.sum for p in points[Meters.LLM_TOKEN_USAGE]
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


def _assert_raw_untraced(response, points, span_exporter):
    assert response.status_code == 200
    assert span_exporter.get_finished_spans() == ()
    assert Meters.LLM_OPERATION_DURATION not in points
    assert Meters.LLM_TOKEN_USAGE not in points


@pytest.mark.parametrize("method, args, kwargs", _RAW_CALLS)
def test_responses_raw_unparseable_body_returns_untraced(
    instrument_legacy, span_exporter, metric_points, method, args, kwargs
):
    raw = _client(_unparseable).responses.with_raw_response
    response = getattr(raw, method)(*args, **kwargs)

    _assert_raw_untraced(response, metric_points(), span_exporter)


@pytest.mark.asyncio
@pytest.mark.parametrize("method, args, kwargs", _RAW_CALLS)
async def test_async_responses_raw_unparseable_body_returns_untraced(
    instrument_legacy, span_exporter, metric_points, method, args, kwargs
):
    raw = _async_client(_unparseable).responses.with_raw_response
    response = await getattr(raw, method)(*args, **kwargs)

    _assert_raw_untraced(response, metric_points(), span_exporter)
