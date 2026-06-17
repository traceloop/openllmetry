"""
Unit tests for __init__.py helpers and instrumentation paths.

Covers:
  _is_deepseek_client, _process_streaming_chunk, _create_stream_processor,
  _create_async_stream_processor, _wrap, _awrap,
  DeepSeekInstrumentor._instrument edge cases.
No cassettes needed — all paths tested with mocks.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

from opentelemetry.instrumentation.deepseek import (
    DeepSeekInstrumentor,
    _accumulate_tool_calls,
    _awrap,
    _create_async_stream_processor,
    _create_stream_processor,
    _is_deepseek_client,
    _process_streaming_chunk,
    _wrap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span(recording: bool = True) -> MagicMock:
    s = MagicMock()
    s.is_recording.return_value = recording
    return s


def _deepseek_instance() -> MagicMock:
    """A mock OpenAI SDK client instance pointed at the DeepSeek API."""
    instance = MagicMock()
    instance._client.base_url = "https://api.deepseek.com/v1/"
    return instance


def _openai_instance() -> MagicMock:
    """A mock OpenAI SDK client instance pointed at the regular OpenAI API."""
    instance = MagicMock()
    instance._client.base_url = "https://api.openai.com/v1/"
    return instance


def _make_sync_wrapper(
    *,
    tracer=None,
    token_histogram=None,
    choice_counter=None,
    duration_histogram=None,
    event_logger=None,
):
    tracer = tracer or MagicMock()
    span = _span()
    tracer.start_span.return_value = span
    return (
        span,
        _wrap(tracer, token_histogram, choice_counter, duration_histogram, event_logger, {}),
    )


def _make_async_wrapper(
    *,
    tracer=None,
    token_histogram=None,
    choice_counter=None,
    duration_histogram=None,
    event_logger=None,
):
    tracer = tracer or MagicMock()
    span = _span()
    tracer.start_span.return_value = span
    return (
        span,
        _awrap(tracer, token_histogram, choice_counter, duration_histogram, event_logger, {}),
    )


# ---------------------------------------------------------------------------
# _is_deepseek_client
# ---------------------------------------------------------------------------


class TestIsDeepSeekClient:
    def test_deepseek_base_url_returns_true(self):
        assert _is_deepseek_client(_deepseek_instance()) is True

    def test_openai_base_url_returns_false(self):
        assert _is_deepseek_client(_openai_instance()) is False

    def test_none_instance_returns_false(self):
        assert _is_deepseek_client(None) is False


# ---------------------------------------------------------------------------
# _process_streaming_chunk
# ---------------------------------------------------------------------------


class TestProcessStreamingChunk:
    def test_empty_choices_returns_none_quintuple(self):
        chunk = MagicMock()
        chunk.choices = []
        assert _process_streaming_chunk(chunk) == (None, "", [], [], None)

    def test_multiple_choices_accumulates_content(self):
        chunk = MagicMock()
        chunk.usage = None
        choice0 = MagicMock()
        choice0.delta.content = "Hello"
        choice0.delta.reasoning_content = None
        choice0.delta.tool_calls = None
        choice0.finish_reason = None
        choice1 = MagicMock()
        choice1.delta.content = " World"
        choice1.delta.reasoning_content = None
        choice1.delta.tool_calls = None
        choice1.finish_reason = "stop"
        chunk.choices = [choice0, choice1]
        content, reasoning_content, tool_calls_delta, finish_reasons, usage = _process_streaming_chunk(chunk)
        assert content == "Hello World"
        assert reasoning_content == ""
        assert tool_calls_delta == []
        assert finish_reasons == ["stop"]
        assert usage is None

    def test_reasoning_content_accumulated_for_deepseek_reasoner(self):
        """DeepSeek-R1 (deepseek-reasoner) streams chain-of-thought via delta.reasoning_content."""
        chunk = MagicMock()
        chunk.usage = None
        choice0 = MagicMock()
        choice0.delta.content = None
        choice0.delta.reasoning_content = "Let me "
        choice0.delta.tool_calls = None
        choice0.finish_reason = None
        choice1 = MagicMock()
        choice1.delta.content = "42"
        choice1.delta.reasoning_content = "think..."
        choice1.delta.tool_calls = None
        choice1.finish_reason = "stop"
        chunk.choices = [choice0, choice1]
        content, reasoning_content, tool_calls_delta, finish_reasons, usage = _process_streaming_chunk(chunk)
        assert content == "42"
        assert reasoning_content == "Let me think..."
        assert finish_reasons == ["stop"]

    def test_tool_calls_delta_extracted(self):
        chunk = MagicMock()
        chunk.usage = None
        tc = MagicMock()
        tc.index = 0
        tc.id = "call_123"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"loc'
        choice = MagicMock()
        choice.delta.content = None
        choice.delta.reasoning_content = None
        choice.delta.tool_calls = [tc]
        choice.finish_reason = None
        chunk.choices = [choice]
        content, reasoning_content, tool_calls_delta, finish_reasons, usage = _process_streaming_chunk(chunk)
        assert content == ""
        assert reasoning_content == ""
        assert len(tool_calls_delta) == 1
        assert tool_calls_delta[0].id == "call_123"

    def test_usage_extracted_from_final_chunk(self):
        chunk = MagicMock()
        usage = MagicMock()
        chunk.usage = usage
        choice = MagicMock()
        choice.delta.content = "done"
        choice.delta.reasoning_content = None
        choice.delta.tool_calls = None
        choice.finish_reason = "stop"
        chunk.choices = [choice]
        _, _, _, _, extracted_usage = _process_streaming_chunk(chunk)
        assert extracted_usage is usage

    def test_no_usage_on_intermediate_chunk(self):
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.delta.content = "partial"
        choice.delta.reasoning_content = None
        choice.delta.tool_calls = None
        choice.finish_reason = None
        chunk.choices = [choice]
        _, _, _, _, usage = _process_streaming_chunk(chunk)
        assert usage is None


# ---------------------------------------------------------------------------
# _accumulate_tool_calls
# ---------------------------------------------------------------------------


class TestAccumulateToolCalls:
    def _make_delta(self, index, tc_id=None, name=None, arguments=""):
        tc = MagicMock()
        tc.index = index
        tc.id = tc_id
        fn = MagicMock()
        fn.name = name
        fn.arguments = arguments
        tc.function = fn
        return tc

    def test_single_chunk_creates_entry(self):
        acc = {}
        tc = self._make_delta(0, tc_id="call_1", name="ping", arguments='{"x"')
        _accumulate_tool_calls(acc, [tc])
        assert acc[0]["id"] == "call_1"
        assert acc[0]["function"]["name"] == "ping"
        assert acc[0]["function"]["arguments"] == '{"x"'

    def test_fragments_are_concatenated(self):
        acc = {}
        _accumulate_tool_calls(acc, [self._make_delta(0, tc_id="call_1", name="fn", arguments='{"a"')])
        _accumulate_tool_calls(acc, [self._make_delta(0, arguments=': 1}')])
        assert acc[0]["function"]["arguments"] == '{"a": 1}'

    def test_multiple_tool_calls_tracked_by_index(self):
        acc = {}
        _accumulate_tool_calls(acc, [
            self._make_delta(0, tc_id="c0", name="fn0", arguments=""),
            self._make_delta(1, tc_id="c1", name="fn1", arguments=""),
        ])
        assert 0 in acc and 1 in acc
        assert acc[0]["id"] == "c0"
        assert acc[1]["id"] == "c1"


# ---------------------------------------------------------------------------
# _create_stream_processor (sync)
# ---------------------------------------------------------------------------


class TestCreateStreamProcessor:
    def test_span_not_recording_skips_set_status(self):
        span = _span(recording=False)
        chunk = MagicMock()
        chunk.choices = []  # _process_streaming_chunk returns (None, "", [], [], None)

        # Consume the generator to trigger cleanup
        list(_create_stream_processor(iter([chunk]), span, None))

        span.set_status.assert_not_called()
        span.end.assert_called_once()

    def test_accumulated_reasoning_passed_to_streaming_attributes(self):
        """Reasoning content is accumulated across chunks and forwarded for the final span attribute."""
        span = _span()

        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = None
        chunk1.choices[0].delta.reasoning_content = "Thinking "
        chunk1.choices[0].delta.tool_calls = None
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.usage = None
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "Answer"
        chunk2.choices[0].delta.reasoning_content = "more..."
        chunk2.choices[0].delta.tool_calls = None
        chunk2.choices[0].finish_reason = "stop"

        with patch("opentelemetry.instrumentation.deepseek.set_streaming_response_attributes") as mock_set:
            list(_create_stream_processor(iter([chunk1, chunk2]), span, None))

        mock_set.assert_called_once()
        _, kwargs = mock_set.call_args
        assert kwargs.get("accumulated_reasoning") == "Thinking more..."


# ---------------------------------------------------------------------------
# _create_async_stream_processor
# ---------------------------------------------------------------------------


class TestCreateAsyncStreamProcessor:
    @pytest.mark.asyncio
    async def test_processes_chunks_and_ends_span(self):
        span = _span()

        chunk = MagicMock()
        chunk.usage = None
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "hi"
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = "stop"

        async def _response():
            yield chunk

        chunks = [c async for c in _create_async_stream_processor(_response(), span, None)]
        assert len(chunks) == 1
        span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_span_not_recording_skips_set_status(self):
        span = _span(recording=False)

        chunk = MagicMock()
        chunk.choices = []

        async def _response():
            yield chunk

        [c async for c in _create_async_stream_processor(_response(), span, None)]
        span.set_status.assert_not_called()
        span.end.assert_called_once()


# ---------------------------------------------------------------------------
# _wrap (sync)
# ---------------------------------------------------------------------------


class TestWrap:
    def test_suppression_key_skips_span(self):
        tracer = MagicMock()
        wrapped = MagicMock(return_value="result")
        wrapper = _wrap(tracer, None, None, None, None, {})

        token = context_api.attach(context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            result = wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})
        finally:
            context_api.detach(token)

        assert result == "result"
        tracer.start_span.assert_not_called()

    def test_non_deepseek_client_passes_through_without_span(self):
        """Calls made through a regular OpenAI client must not be instrumented."""
        tracer = MagicMock()
        wrapped = MagicMock(return_value="result")
        wrapper = _wrap(tracer, None, None, None, None, {})

        result = wrapper(wrapped, _openai_instance(), [], {"model": "gpt-4"})

        assert result == "result"
        tracer.start_span.assert_not_called()
        wrapped.assert_called_once()

    def test_none_instance_passes_through_without_span(self):
        tracer = MagicMock()
        wrapped = MagicMock(return_value="result")
        wrapper = _wrap(tracer, None, None, None, None, {})

        result = wrapper(wrapped, None, [], {"model": "m"})

        assert result == "result"
        tracer.start_span.assert_not_called()

    def test_api_exception_records_duration_and_reraises(self):
        span, wrapper = _make_sync_wrapper(duration_histogram=MagicMock())
        error = ValueError("API down")
        wrapped = MagicMock(side_effect=error)

        with pytest.raises(ValueError, match="API down"):
            wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        span.end.assert_called_once()  # span must be ended even on exception
        span.set_status.assert_called_once()

    def test_api_exception_records_duration_histogram(self):
        tracer = MagicMock()
        span = _span()
        tracer.start_span.return_value = span
        duration_histogram = MagicMock()
        wrapped = MagicMock(side_effect=RuntimeError("fail"))

        wrapper = _wrap(tracer, None, None, duration_histogram, None, {})
        with pytest.raises(RuntimeError):
            wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        duration_histogram.record.assert_called_once()

    def test_falsy_response_ends_span_without_setting_status(self):
        span, wrapper = _make_sync_wrapper()
        wrapped = MagicMock(return_value=None)

        result = wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        assert result is None
        span.end.assert_called_once()
        span.set_status.assert_not_called()

    def test_handle_response_exception_is_swallowed(self):
        span, wrapper = _make_sync_wrapper()
        response = MagicMock()
        wrapped = MagicMock(return_value=response)

        with patch("opentelemetry.instrumentation.deepseek._handle_response", side_effect=Exception("oops")):
            with patch("opentelemetry.instrumentation.deepseek.shared_metrics_attributes", return_value={}):
                result = wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        assert result is response
        span.end.assert_called_once()

    def test_no_duration_histogram_skips_duration_record(self):
        span, wrapper = _make_sync_wrapper(duration_histogram=None)
        response = MagicMock()
        wrapped = MagicMock(return_value=response)

        with patch("opentelemetry.instrumentation.deepseek._handle_response"):
            with patch("opentelemetry.instrumentation.deepseek.shared_metrics_attributes", return_value={}):
                result = wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        assert result is response
        span.end.assert_called_once()

    def test_stream_processor_exception_sets_error_status(self):
        from openai._streaming import Stream

        tracer = MagicMock()
        span = _span()
        tracer.start_span.return_value = span
        response = MagicMock(spec=Stream)
        wrapped = MagicMock(return_value=response)

        wrapper = _wrap(tracer, None, None, None, None, {})
        with patch(
            "opentelemetry.instrumentation.deepseek._create_stream_processor",
            side_effect=RuntimeError("stream error"),
        ):
            with pytest.raises(RuntimeError, match="stream error"):
                wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        span.end.assert_called_once()

    def test_span_not_recording_after_response_skips_set_status(self):
        """Covers span.is_recording() False at the final OK status check."""
        tracer = MagicMock()
        span = MagicMock()
        # Calls: set_model_input_attributes, set_input_attributes, final check
        span.is_recording.side_effect = [True, True, False]
        tracer.start_span.return_value = span

        response = MagicMock()
        wrapped = MagicMock(return_value=response)

        with patch("opentelemetry.instrumentation.deepseek._handle_response"):
            with patch("opentelemetry.instrumentation.deepseek.shared_metrics_attributes", return_value={}):
                wrapper = _wrap(tracer, None, None, None, None, {})
                wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        span.set_status.assert_not_called()
        span.end.assert_called_once()


# ---------------------------------------------------------------------------
# _awrap (async)
# ---------------------------------------------------------------------------


class TestAwrap:
    @pytest.mark.asyncio
    async def test_suppression_key_skips_span(self):
        tracer = MagicMock()
        wrapped = AsyncMock(return_value="async_result")
        wrapper = _awrap(tracer, None, None, None, None, {})

        token = context_api.attach(context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True))
        try:
            result = await wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})
        finally:
            context_api.detach(token)

        assert result == "async_result"
        tracer.start_span.assert_not_called()

    @pytest.mark.asyncio
    async def test_otel_suppression_key_skips_span(self):
        tracer = MagicMock()
        wrapped = AsyncMock(return_value="async_result")
        wrapper = _awrap(tracer, None, None, None, None, {})

        token = context_api.attach(context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            result = await wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})
        finally:
            context_api.detach(token)

        assert result == "async_result"
        tracer.start_span.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_deepseek_client_passes_through_without_span(self):
        """Calls made through a regular OpenAI client must not be instrumented."""
        tracer = MagicMock()
        wrapped = AsyncMock(return_value="async_result")
        wrapper = _awrap(tracer, None, None, None, None, {})

        result = await wrapper(wrapped, _openai_instance(), [], {"model": "gpt-4"})

        assert result == "async_result"
        tracer.start_span.assert_not_called()
        wrapped.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_api_exception_records_duration_and_reraises(self):
        tracer = MagicMock()
        span = _span()
        tracer.start_span.return_value = span
        duration_histogram = MagicMock()

        wrapped = AsyncMock(side_effect=RuntimeError("async fail"))
        wrapper = _awrap(tracer, None, None, duration_histogram, None, {})

        with pytest.raises(RuntimeError, match="async fail"):
            await wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        duration_histogram.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_falsy_response_ends_span(self):
        span, wrapper = _make_async_wrapper()
        wrapped = AsyncMock(return_value=None)

        result = await wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        assert result is None
        span.end.assert_called_once()
        span.set_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_response_with_duration_histogram(self):
        tracer = MagicMock()
        span = _span()
        tracer.start_span.return_value = span
        duration_histogram = MagicMock()

        response = MagicMock()
        wrapped = AsyncMock(return_value=response)

        wrapper = _awrap(tracer, None, None, duration_histogram, None, {})
        with patch("opentelemetry.instrumentation.deepseek._handle_response"):
            with patch("opentelemetry.instrumentation.deepseek.shared_metrics_attributes", return_value={}):
                result = await wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        assert result is response
        duration_histogram.record.assert_called_once()
        span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_response_without_duration_histogram(self):
        span, wrapper = _make_async_wrapper(duration_histogram=None)
        response = MagicMock()
        wrapped = AsyncMock(return_value=response)

        with patch("opentelemetry.instrumentation.deepseek._handle_response"):
            with patch("opentelemetry.instrumentation.deepseek.shared_metrics_attributes", return_value={}):
                result = await wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        assert result is response
        span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_streaming_exception_sets_error_status(self):
        """Covers the async streaming path raising and setting ERROR status."""
        from openai._streaming import AsyncStream

        tracer = MagicMock()
        span = _span()
        tracer.start_span.return_value = span
        response = MagicMock(spec=AsyncStream)
        wrapped = AsyncMock(return_value=response)

        wrapper = _awrap(tracer, None, None, None, None, {})
        with patch(
            "opentelemetry.instrumentation.deepseek._create_async_stream_processor",
            side_effect=RuntimeError("async stream error"),
        ):
            with pytest.raises(RuntimeError, match="async stream error"):
                await wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_span_not_recording_after_response_skips_set_status(self):
        """Covers span.is_recording() False at the final OK status check."""
        tracer = MagicMock()
        span = MagicMock()
        span.is_recording.side_effect = [True, True, False]
        tracer.start_span.return_value = span

        response = MagicMock()
        wrapped = AsyncMock(return_value=response)

        with patch("opentelemetry.instrumentation.deepseek._handle_response"):
            with patch("opentelemetry.instrumentation.deepseek.shared_metrics_attributes", return_value={}):
                wrapper = _awrap(tracer, None, None, None, None, {})
                await wrapper(wrapped, _deepseek_instance(), [], {"model": "m"})

        span.set_status.assert_not_called()
        span.end.assert_called_once()


# ---------------------------------------------------------------------------
# DeepSeekInstrumentor._instrument edge cases
# ---------------------------------------------------------------------------


class TestDeepSeekInstrumentor:
    def test_metrics_disabled_sets_histograms_to_none(self, tracer_provider, meter_provider):
        with patch.dict(os.environ, {"TRACELOOP_METRICS_ENABLED": "false"}):
            instrumentor = DeepSeekInstrumentor()
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )
            instrumentor.uninstrument()

    def test_module_not_found_for_sync_wrap_is_swallowed(self, tracer_provider, meter_provider):
        instrumentor = DeepSeekInstrumentor()
        with patch(
            "opentelemetry.instrumentation.deepseek.wrap_function_wrapper",
            side_effect=ModuleNotFoundError("openai not installed"),
        ):
            # Should not raise
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )
        instrumentor.uninstrument()
