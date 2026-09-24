"""Unit tests for streaming metrics in the Groq instrumentation.

Covers the fix for #4419: streaming calls now record token usage and
duration metrics once the stream is fully drained.
All tests use fake streams — no network calls, no cassettes.
"""

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.groq import (
    _create_async_stream_processor,
    _create_stream_processor,
)
from opentelemetry.instrumentation.groq.utils import streaming_metrics_attributes
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv_ai import Meters
from opentelemetry.trace import get_tracer


def _chunk(content="", finish_reason=None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        x_groq=SimpleNamespace(usage=usage) if usage else None,
    )


def _usage(prompt=10, completion=5):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class _FailingAsyncStream:
    """Async stream that yields one chunk then raises."""

    async def __aiter__(self):
        yield _chunk(content="hello")
        raise RuntimeError("async stream exploded")


def _find_metric(metrics_data, name):
    """Return the first metric whose instrument name matches, or None."""
    if metrics_data is None:
        return None
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == name:
                    return metric
    return None


def _find_data_point(metric, attributes_subset):
    """Return the first data point whose attributes contain the given subset."""
    for dp in metric.data.data_points:
        dp_attrs = dict(dp.attributes)
        if all(dp_attrs.get(k) == v for k, v in attributes_subset.items()):
            return dp
    return None


# ---------------------------------------------------------------------------
# streaming_metrics_attributes
# ---------------------------------------------------------------------------


class TestStreamingMetricsAttributes:
    def test_returns_provider_and_model(self):
        attrs = streaming_metrics_attributes("llama-3.3-70b-versatile")
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "groq"
        assert attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# _create_stream_processor (sync)
# ---------------------------------------------------------------------------


class TestStreamProcessorMetrics:
    def test_records_token_and_duration_metrics(self, reader, tracer_provider, meter_provider):
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        token_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_TOKEN_USAGE)
        duration_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_OPERATION_DURATION)

        usage = _usage(prompt=10, completion=5)
        stream = _FakeStream(
            [
                _chunk(content="hello"),
                _chunk(content=" world", finish_reason="stop", usage=usage),
            ]
        )

        processor = _create_stream_processor(
            stream,
            span,
            None,
            token_histogram,
            duration_histogram,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )
        for _ in processor:
            pass

        metrics_data = reader.get_metrics_data()

        token_metric = _find_metric(metrics_data, Meters.LLM_TOKEN_USAGE)
        assert token_metric is not None
        input_dp = _find_data_point(token_metric, {GenAIAttributes.GEN_AI_TOKEN_TYPE: "input"})
        assert input_dp is not None and input_dp.sum == 10
        output_dp = _find_data_point(token_metric, {GenAIAttributes.GEN_AI_TOKEN_TYPE: "output"})
        assert output_dp is not None and output_dp.sum == 5

        duration_metric = _find_metric(metrics_data, Meters.LLM_OPERATION_DURATION)
        assert duration_metric is not None
        assert duration_metric.data.data_points[0].sum > 0

    def test_records_duration_even_without_usage(self, reader, tracer_provider, meter_provider):
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        duration_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_OPERATION_DURATION)

        stream = _FakeStream([_chunk(content="hello", finish_reason="stop")])

        processor = _create_stream_processor(
            stream,
            span,
            None,
            token_histogram=None,
            duration_histogram=duration_histogram,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )
        for _ in processor:
            pass

        metrics_data = reader.get_metrics_data()
        duration_metric = _find_metric(metrics_data, Meters.LLM_OPERATION_DURATION)
        assert duration_metric is not None
        assert duration_metric.data.data_points[0].sum > 0

        token_metric = _find_metric(metrics_data, Meters.LLM_TOKEN_USAGE)
        assert token_metric is None

    def test_metrics_disabled_skips_recording(self, reader, tracer_provider, meter_provider):
        """Histograms are None when metrics are disabled; recording must be skipped."""
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        stream = _FakeStream([_chunk(content="hello", finish_reason="stop", usage=_usage())])

        processor = _create_stream_processor(
            stream,
            span,
            None,
            token_histogram=None,
            duration_histogram=None,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )
        for _ in processor:
            pass

        metrics_data = reader.get_metrics_data()
        assert _find_metric(metrics_data, Meters.LLM_TOKEN_USAGE) is None
        assert _find_metric(metrics_data, Meters.LLM_OPERATION_DURATION) is None


# ---------------------------------------------------------------------------
# _create_async_stream_processor
# ---------------------------------------------------------------------------


class TestAsyncStreamProcessorMetrics:
    @pytest.mark.asyncio
    async def test_async_records_token_and_duration_metrics(self, reader, tracer_provider, meter_provider):
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        token_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_TOKEN_USAGE)
        duration_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_OPERATION_DURATION)

        usage = _usage(prompt=7, completion=3)
        stream = _FakeAsyncStream(
            [
                _chunk(content="hello"),
                _chunk(content=" world", finish_reason="stop", usage=usage),
            ]
        )

        processor = _create_async_stream_processor(
            stream,
            span,
            None,
            token_histogram,
            duration_histogram,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )
        async for _ in processor:
            pass

        metrics_data = reader.get_metrics_data()

        token_metric = _find_metric(metrics_data, Meters.LLM_TOKEN_USAGE)
        assert token_metric is not None
        input_dp = _find_data_point(token_metric, {GenAIAttributes.GEN_AI_TOKEN_TYPE: "input"})
        assert input_dp is not None and input_dp.sum == 7
        output_dp = _find_data_point(token_metric, {GenAIAttributes.GEN_AI_TOKEN_TYPE: "output"})
        assert output_dp is not None and output_dp.sum == 3

        duration_metric = _find_metric(metrics_data, Meters.LLM_OPERATION_DURATION)
        assert duration_metric is not None
        assert duration_metric.data.data_points[0].sum > 0


# ---------------------------------------------------------------------------
# _create_stream_processor failure paths (review: record duration on error)
# ---------------------------------------------------------------------------


class TestStreamProcessorErrorMetrics:
    def test_sync_records_duration_metric_on_stream_failure(self, reader, tracer_provider, meter_provider):
        """A stream that raises mid-iteration must still emit an operation-duration
        metric (with error attributes) before re-raising."""
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        duration_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_OPERATION_DURATION)

        def _failing():
            yield _chunk(content="hello")
            raise RuntimeError("stream exploded")

        processor = _create_stream_processor(
            _FakeStream(_failing()),
            span,
            None,
            token_histogram=None,
            duration_histogram=duration_histogram,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )
        with pytest.raises(RuntimeError):
            for _ in processor:
                pass

        metrics_data = reader.get_metrics_data()
        duration_metric = _find_metric(metrics_data, Meters.LLM_OPERATION_DURATION)
        assert duration_metric is not None
        assert duration_metric.data.data_points[0].sum > 0
        dp = duration_metric.data.data_points[0]
        assert dict(dp.attributes)["error.type"] == "RuntimeError"

    def test_sync_skips_duration_metric_when_disabled_on_failure(self, reader, tracer_provider, meter_provider):
        """With histograms disabled (None), a failing stream must not crash and
        must still propagate the exception."""
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")

        def _failing():
            yield _chunk(content="hello")
            raise ValueError("boom")

        processor = _create_stream_processor(
            _FakeStream(_failing()),
            span,
            None,
            token_histogram=None,
            duration_histogram=None,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )
        with pytest.raises(ValueError):
            for _ in processor:
                pass

        metrics_data = reader.get_metrics_data()
        assert _find_metric(metrics_data, Meters.LLM_OPERATION_DURATION) is None

    @pytest.mark.asyncio
    async def test_async_records_duration_metric_on_stream_failure(self, reader, tracer_provider, meter_provider):
        """The async stream processor must emit the duration metric with error
        attributes when iteration fails."""
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        duration_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_OPERATION_DURATION)

        processor = _create_async_stream_processor(
            _FailingAsyncStream(),
            span,
            None,
            token_histogram=None,
            duration_histogram=duration_histogram,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )
        with pytest.raises(RuntimeError):
            async for _ in processor:
                pass

        metrics_data = reader.get_metrics_data()
        duration_metric = _find_metric(metrics_data, Meters.LLM_OPERATION_DURATION)
        assert duration_metric is not None
        assert duration_metric.data.data_points[0].sum > 0
        dp = duration_metric.data.data_points[0]
        assert dict(dp.attributes)["error.type"] == "RuntimeError"
