import asyncio
import time

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import Meters
from opentelemetry.trace import StatusCode


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


@pytest.fixture
def poll_interval(vcr):
    # Only a live recording has to wait for the background response; a replayed cassette doesn't.
    return 0 if vcr.write_protected else 3


def _assert_one_duration(points, span_exporter, finished_ns, response_model):
    [span] = span_exporter.get_finished_spans()
    assert span.status.status_code != StatusCode.ERROR
    [duration] = points[Meters.LLM_OPERATION_DURATION]
    assert duration.count == 1
    # Recorded after the span ended and before the call returned, measured from the create's start.
    assert (span.end_time - span.start_time) / 1e9 <= duration.sum <= (finished_ns - span.start_time) / 1e9
    assert ERROR_TYPE not in duration.attributes
    assert duration.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert duration.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == response_model
    assert duration.attributes["server.address"] == "https://api.openai.com/v1/"
    assert duration.attributes["stream"] is False


def _tokens(points):
    return {p.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]: p.sum for p in points.get(Meters.LLM_TOKEN_USAGE, [])}


@pytest.mark.vcr
def test_background_poll_records_one_duration_and_final_usage(
    instrument_legacy, span_exporter, metric_points, openai_client, poll_interval
):
    response = openai_client.responses.create(
        model="gpt-6-luna", input="Say hello in one word.", background=True, max_output_tokens=16
    )
    assert response.status in ("queued", "in_progress")
    assert response.usage is None
    for _ in range(5):
        time.sleep(poll_interval)
        response = openai_client.responses.retrieve(response.id)
        if response.status not in ("queued", "in_progress"):
            break

    assert response.status in ("completed", "incomplete")
    points = metric_points()
    _assert_one_duration(points, span_exporter, time.time_ns(), response.model)
    assert _tokens(points) == {"input": response.usage.input_tokens, "output": response.usage.output_tokens}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_background_poll_records_one_duration_and_final_usage(
    instrument_legacy, span_exporter, metric_points, async_openai_client, poll_interval
):
    response = await async_openai_client.responses.create(
        model="gpt-6-luna", input="Say hello in one word.", background=True, max_output_tokens=16
    )
    assert response.status in ("queued", "in_progress")
    assert response.usage is None
    for _ in range(5):
        await asyncio.sleep(poll_interval)
        response = await async_openai_client.responses.retrieve(response.id)
        if response.status not in ("queued", "in_progress"):
            break

    assert response.status in ("completed", "incomplete")
    points = metric_points()
    _assert_one_duration(points, span_exporter, time.time_ns(), response.model)
    assert _tokens(points) == {"input": response.usage.input_tokens, "output": response.usage.output_tokens}


@pytest.mark.vcr
def test_background_cancel_records_one_duration(instrument_legacy, span_exporter, metric_points, openai_client):
    created = openai_client.responses.create(
        model="gpt-6-luna", input="Write a long story about the sea.", background=True, max_output_tokens=1024
    )
    cancelled = openai_client.responses.cancel(created.id)

    assert cancelled.id == created.id
    assert cancelled.status == "cancelled"
    points = metric_points()
    _assert_one_duration(points, span_exporter, time.time_ns(), created.model)
    assert _tokens(points) == {}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_background_cancel_records_one_duration(
    instrument_legacy, span_exporter, metric_points, async_openai_client
):
    created = await async_openai_client.responses.create(
        model="gpt-6-luna", input="Write a long story about the sea.", background=True, max_output_tokens=1024
    )
    cancelled = await async_openai_client.responses.cancel(created.id)

    assert cancelled.id == created.id
    assert cancelled.status == "cancelled"
    points = metric_points()
    _assert_one_duration(points, span_exporter, time.time_ns(), created.model)
    assert _tokens(points) == {}
