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
    _record_streaming_metrics,
)
from opentelemetry.instrumentation.groq.utils import streaming_metrics_attributes
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv_ai import Meters
from opentelemetry.trace import get_tracer


def _chunk(content="", finish_reason=None, usage=None, model=None):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        x_groq=SimpleNamespace(usage=usage) if usage else None,
    )


def _empty_choice_usage_chunk(usage):
    """A trailing chunk with no choices that still carries the usage payload, as Groq sends it."""
    chunk = _chunk(usage=usage)
    chunk.choices = []
    return chunk


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

    def test_records_token_metrics_from_a_choice_less_final_chunk(self, reader, tracer_provider, meter_provider):
        """Usage that arrives on a trailing chunk with no choices must still be recorded.

        Groq attaches the usage payload to a final chunk that carries no choices, so the
        empty-choices guard in `_process_streaming_chunk` must not drop it: this is the
        difference between an empty token histogram and a populated one.
        """
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        token_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_TOKEN_USAGE)

        stream = _FakeStream(
            [
                _chunk(content="hello"),
                _chunk(content=" world", finish_reason="stop"),
                _empty_choice_usage_chunk(_usage(prompt=10, completion=5)),
            ]
        )

        for _ in _create_stream_processor(
            stream,
            span,
            None,
            token_histogram,
            None,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        ):
            pass

        token_metric = _find_metric(reader.get_metrics_data(), Meters.LLM_TOKEN_USAGE)
        assert token_metric is not None
        input_dp = _find_data_point(token_metric, {GenAIAttributes.GEN_AI_TOKEN_TYPE: "input"})
        output_dp = _find_data_point(token_metric, {GenAIAttributes.GEN_AI_TOKEN_TYPE: "output"})
        assert input_dp is not None and input_dp.sum == 10
        assert output_dp is not None and output_dp.sum == 5

    def test_token_points_carry_non_streaming_attribute_set(self, reader, tracer_provider, meter_provider):
        """Token points must match the non-streaming attribute keys (semconv requires operation.name)."""
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        token_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_TOKEN_USAGE)

        stream = _FakeStream([_chunk(content="hello", finish_reason="stop", usage=_usage())])
        for _ in _create_stream_processor(
            stream,
            span,
            None,
            token_histogram,
            None,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        ):
            pass

        metrics_data = reader.get_metrics_data()
        token_metric = _find_metric(metrics_data, Meters.LLM_TOKEN_USAGE)
        assert token_metric is not None
        assert {frozenset(dict(dp.attributes)) for dp in token_metric.data.data_points} == {
            frozenset(
                {
                    GenAIAttributes.GEN_AI_PROVIDER_NAME,
                    GenAIAttributes.GEN_AI_OPERATION_NAME,
                    GenAIAttributes.GEN_AI_REQUEST_MODEL,
                    GenAIAttributes.GEN_AI_RESPONSE_MODEL,
                    GenAIAttributes.GEN_AI_TOKEN_TYPE,
                }
            )
        }

    def test_uses_last_chunk_model_as_response_model(self, reader, tracer_provider, meter_provider):
        """gen_ai.response.model comes from the server's chunks, not the request kwargs."""
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        token_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_TOKEN_USAGE)
        duration_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_OPERATION_DURATION)

        stream = _FakeStream(
            [
                _chunk(content="hello", model="server-reported-model"),
                _chunk(content=" world", finish_reason="stop", usage=_usage(), model="server-reported-model"),
            ]
        )
        for _ in _create_stream_processor(
            stream,
            span,
            None,
            token_histogram,
            duration_histogram,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        ):
            pass

        metrics_data = reader.get_metrics_data()
        token_dp = _find_metric(metrics_data, Meters.LLM_TOKEN_USAGE).data.data_points[0]
        duration_dp = _find_metric(metrics_data, Meters.LLM_OPERATION_DURATION).data.data_points[0]
        # Duration points keep the non-streaming attribute set (provider + response
        # model only); the request model is added to token points alone.
        assert dict(duration_dp.attributes)[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "server-reported-model"
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL not in dict(duration_dp.attributes)

        token_attrs = dict(token_dp.attributes)
        assert token_attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "server-reported-model"
        assert token_attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "llama-3.3-70b-versatile"

    def test_falls_back_to_request_model_when_chunks_carry_none(
        self, reader, tracer_provider, meter_provider
    ):
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        token_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_TOKEN_USAGE)

        stream = _FakeStream([_chunk(content="hello", finish_reason="stop", usage=_usage())])
        for _ in _create_stream_processor(
            stream,
            span,
            None,
            token_histogram,
            None,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        ):
            pass

        metrics_data = reader.get_metrics_data()
        token_dp = _find_metric(metrics_data, Meters.LLM_TOKEN_USAGE).data.data_points[0]
        assert dict(token_dp.attributes)[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "llama-3.3-70b-versatile"

    def test_recording_failure_never_reaches_the_caller(self):
        """_record_streaming_metrics runs in the generator's else block — it must not raise."""

        class _Boom:
            def record(self, *args, **kwargs):
                raise RuntimeError("histogram exploded")

        _record_streaming_metrics(_usage(), _Boom(), _Boom(), 0, "llama-3.3-70b-versatile")

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

    @pytest.mark.asyncio
    async def test_async_records_token_metrics_from_a_choice_less_final_chunk(
        self, reader, tracer_provider, meter_provider
    ):
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")
        token_histogram = meter_provider.get_meter("test").create_histogram(name=Meters.LLM_TOKEN_USAGE)

        stream = _FakeAsyncStream(
            [
                _chunk(content="hello"),
                _chunk(content=" world", finish_reason="stop"),
                _empty_choice_usage_chunk(_usage(prompt=7, completion=3)),
            ]
        )

        processor = _create_async_stream_processor(
            stream,
            span,
            None,
            token_histogram,
            None,
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )
        async for _ in processor:
            pass

        token_metric = _find_metric(reader.get_metrics_data(), Meters.LLM_TOKEN_USAGE)
        assert token_metric is not None
        input_dp = _find_data_point(token_metric, {GenAIAttributes.GEN_AI_TOKEN_TYPE: "input"})
        output_dp = _find_data_point(token_metric, {GenAIAttributes.GEN_AI_TOKEN_TYPE: "output"})
        assert input_dp is not None and input_dp.sum == 7
        assert output_dp is not None and output_dp.sum == 3


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


class _RaisingHistogram:
    """A histogram whose record() blows up, as the SDK's can under a broken exporter."""

    def record(self, *args, **kwargs):
        raise RuntimeError("histogram exploded")


class TestTelemetryFailureDoesNotMaskCallerError:
    """Recording a metric must never change which exception the caller sees."""

    def test_sync_stream_error_survives_a_broken_histogram(self, tracer_provider):
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")

        def _failing():
            yield _chunk(content="hello")
            raise RuntimeError("stream exploded")

        processor = _create_stream_processor(
            _FakeStream(_failing()),
            span,
            None,
            token_histogram=None,
            duration_histogram=_RaisingHistogram(),
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )

        with pytest.raises(RuntimeError, match="stream exploded"):
            for _ in processor:
                pass

    @pytest.mark.asyncio
    async def test_async_stream_error_survives_a_broken_histogram(self, tracer_provider):
        span = get_tracer("test", tracer_provider=tracer_provider).start_span("chat llama-3.3-70b-versatile")

        async def _failing():
            yield _chunk(content="hello")
            raise RuntimeError("async stream exploded")

        processor = _create_async_stream_processor(
            _failing(),
            span,
            None,
            token_histogram=None,
            duration_histogram=_RaisingHistogram(),
            start_time=0,
            llm_model="llama-3.3-70b-versatile",
        )

        with pytest.raises(RuntimeError, match="async stream exploded"):
            async for _ in processor:
                pass
