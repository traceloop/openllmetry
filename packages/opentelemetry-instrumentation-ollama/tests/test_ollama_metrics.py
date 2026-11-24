import json
from unittest.mock import AsyncMock, Mock

import httpx
import ollama
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import (
    gen_ai_metrics as GenAIMetrics,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv_ai import Meters
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace.status import StatusCode
from opentelemetry.instrumentation.ollama import _awrap, _wrap
from opentelemetry.instrumentation.ollama.config import Config


def _collect_metrics(reader):
    """Helper to flatten all metrics data points."""
    data = reader.get_metrics_data()
    points = []
    if data is None:
        return points
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for dp in metric.data.data_points:
                    points.append((metric.name, dp))
    return points


def _mock_operation_response(status, stream):
    def respond(request):
        if status != 200:
            return httpx.Response(status, json={"error": "mock failure"})
        response = {
            "model": "test-model",
            "done": True,
            "message": {"role": "assistant", "content": "Hello"},
            "response": "Hello",
            "prompt_eval_count": 3,
            "eval_count": 1,
        }
        if stream:
            return httpx.Response(200, content=json.dumps(response) + "\n")
        return httpx.Response(200, json=response)

    return httpx.MockTransport(respond)


def _assert_operation_duration(reader, span_exporter, failed):
    points = [dp for name, dp in _collect_metrics(reader) if name == Meters.LLM_OPERATION_DURATION]
    assert sum(dp.count for dp in points) == 1
    assert len(points) == 1
    point = points[0]
    assert point.sum >= 0
    assert point.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "Ollama"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    exceptions = [event for event in span.events if event.name == "exception"]
    if failed:
        assert point.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "test-model"
        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in point.attributes
        assert point.attributes[ERROR_TYPE] == "ResponseError"
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes[ERROR_TYPE] == "ResponseError"
        assert "mock failure" in span.status.description
        assert len(exceptions) == 1
        assert "mock failure" in exceptions[0].attributes["exception.message"]
    else:
        assert point.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "test-model"
        assert ERROR_TYPE not in point.attributes
        assert span.status.status_code != StatusCode.ERROR
        assert not exceptions


@pytest.mark.parametrize("method", ["chat", "generate"])
@pytest.mark.parametrize("status,stream", [(200, False), (500, False), (200, True)])
def test_ollama_operation_duration_once(instrument_legacy, reader, span_exporter, method, status, stream):
    client = ollama.Client(host="http://mock", transport=_mock_operation_response(status, stream))
    try:
        if status == 500:
            with pytest.raises(ollama.ResponseError, match="mock failure") as error:
                getattr(client, method)(model="test-model", stream=stream)
            assert error.value.status_code == 500
        else:
            response = getattr(client, method)(model="test-model", stream=stream)
            if stream:
                list(response)
    finally:
        client._client.close()
    _assert_operation_duration(reader, span_exporter, status == 500)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat", "generate"])
@pytest.mark.parametrize("status,stream", [(200, False), (500, False), (200, True)])
async def test_async_ollama_operation_duration_once(instrument_legacy, reader, span_exporter, method, status, stream):
    client = ollama.AsyncClient(host="http://mock", transport=_mock_operation_response(status, stream))
    try:
        if status == 500:
            with pytest.raises(ollama.ResponseError, match="mock failure") as error:
                await getattr(client, method)(model="test-model", stream=stream)
            assert error.value.status_code == 500
        else:
            response = await getattr(client, method)(model="test-model", stream=stream)
            if stream:
                async for _ in response:
                    pass
    finally:
        await client._client.aclose()
    _assert_operation_duration(reader, span_exporter, status == 500)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("logger_fails", [False, True])
async def test_duration_recording_failure_preserves_request_error(
    tracer_provider, span_exporter, monkeypatch, is_async, logger_fails
):
    original_error = ollama.ResponseError("original request failure", 500)
    wrapped = AsyncMock(side_effect=original_error) if is_async else Mock(side_effect=original_error)
    histogram = Mock()
    histogram.record.side_effect = RuntimeError("metric failure")
    exception_logger = Mock(side_effect=RuntimeError("logger failure")) if logger_fails else None
    monkeypatch.setattr(Config, "exception_logger", exception_logger)
    wrapper = (_awrap if is_async else _wrap)(
        tracer_provider.get_tracer(__name__),
        None,
        histogram,
        None,
        None,
        None,
        {"method": "chat", "span_name": "ollama.chat"},
    )

    with pytest.raises(ollama.ResponseError) as error:
        result = wrapper(wrapped, None, (), {"json": {"model": "test-model", "messages": []}})
        if is_async:
            await result

    assert error.value is original_error
    histogram.record.assert_called_once()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes[ERROR_TYPE] == "ResponseError"
    exceptions = [event for event in spans[0].events if event.name == "exception"]
    assert len(exceptions) == 1
    assert "original request failure" in exceptions[0].attributes["exception.message"]


@pytest.mark.vcr
def test_ollama_streaming_metrics(instrument_legacy, reader):
    gen = ollama.generate(
        model="gemma3:1b",
        prompt="Tell me a joke about OpenTelemetry",
        stream=True,
    )
    for _ in gen:
        pass

    points = _collect_metrics(reader)
    # Assert metrics for token usage, operation duration, time to first token,
    # and streaming time to generate are present
    assert any(name == Meters.LLM_TOKEN_USAGE for name, _ in points), "Token usage metric not found"
    assert any(name == Meters.LLM_OPERATION_DURATION for name, _ in points), "Operation duration metric not found"
    assert any(name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN for name, _ in points), \
        "Time to first token metric not found"
    assert any(name == Meters.LLM_STREAMING_TIME_TO_GENERATE for name, _ in points), \
        "Streaming time to generate metric not found"

    # Further assert that time-to-first-token is greater than 0 and has the system attribute
    for name, dp in points:
        if name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN:
            assert dp.sum > 0, "Time to first token should be greater than 0"
            assert dp.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Ollama"
            break


@pytest.mark.vcr
def test_ollama_streaming_time_to_generate_metrics(instrument_legacy, reader):
    gen = ollama.generate(
        model="gemma3:1b",
        prompt="Tell me a joke about OpenTelemetry",
        stream=True,
    )
    for _ in gen:
        pass

    points = _collect_metrics(reader)
    # Assert metrics for streaming time to generate is present
    assert any(name == Meters.LLM_STREAMING_TIME_TO_GENERATE for name, _ in points), \
        "Streaming time to generate metric not found"

    # Further assert that streaming-time-to-generate is greater than 0 and has the system attribute
    for name, dp in points:
        if name == Meters.LLM_STREAMING_TIME_TO_GENERATE:
            assert dp.sum > 0, "Streaming time to generate should be greater than 0"
            assert dp.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Ollama"
            assert dp.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL) is not None
            break


@pytest.mark.vcr
def test_ollama_operation_duration_includes_model_attribute(instrument_legacy, reader):
    """Test that LLM_OPERATION_DURATION metric includes gen_ai.response.model attribute."""
    ollama.chat(
        model="gemma3:1b",
        messages=[
            {"role": "user", "content": "Hello, this is a test for model attribute."},
        ],
    )

    points = _collect_metrics(reader)

    operation_duration_found = False
    model_attribute_found = False

    for name, dp in points:
        if name == Meters.LLM_OPERATION_DURATION:
            operation_duration_found = True
            # Check that the metric has both required attributes
            assert dp.attributes.get(SpanAttributes.LLM_SYSTEM) == "Ollama", \
                "LLM_OPERATION_DURATION should have gen_ai.system attribute"

            model_name = dp.attributes.get(SpanAttributes.LLM_RESPONSE_MODEL)
            if model_name is not None:
                model_attribute_found = True
                assert model_name == "gemma3:1b", \
                    f"Expected model 'gemma3:1b', but got '{model_name}'"

            assert dp.sum > 0, "Operation duration should be greater than 0"
            break

    assert operation_duration_found, "LLM_OPERATION_DURATION metric not found"
    assert model_attribute_found, "gen_ai.response.model attribute not found in LLM_OPERATION_DURATION metric"
