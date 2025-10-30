import time
from unittest.mock import Mock, patch
from uuid import uuid4
from langchain_core.outputs import LLMResult, Generation
from opentelemetry.instrumentation.langchain.callback_handler import TraceloopCallbackHandler
from opentelemetry.instrumentation.langchain.span_utils import SpanHolder
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span


class TestStreamingMetrics:
    """Test the new streaming metrics functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = Mock()
        self.duration_histogram = Mock()
        self.token_histogram = Mock()
        self.ttft_histogram = Mock()
        self.streaming_time_histogram = Mock()
        self.choices_counter = Mock()
        self.exception_counter = Mock()

        self.handler = TraceloopCallbackHandler(
            self.tracer,
            self.duration_histogram,
            self.token_histogram,
            self.ttft_histogram,
            self.streaming_time_histogram,
            self.choices_counter,
            self.exception_counter,
        )

    def test_ttft_metric_recorded_on_first_token(self):
        """Test that TTFT metric is recorded when first token arrives."""
        run_id = uuid4()
        mock_span = Mock(spec=Span)
        mock_span.attributes = {SpanAttributes.LLM_SYSTEM: "Langchain"}

        # Use mock time for stable testing
        with patch('opentelemetry.instrumentation.langchain.callback_handler.time.time') as mock_time, \
             patch('opentelemetry.instrumentation.langchain.span_utils.time.time') as mock_span_time:

            start_time = 1000.0
            mock_time.return_value = start_time
            mock_span_time.return_value = start_time

            span_holder = SpanHolder(
                span=mock_span,
                token=None,
                context=None,
                children=[],
                workflow_name="test",
                entity_name="test",
                entity_path="test",
                start_time=start_time
            )
            self.handler.spans[run_id] = span_holder

            mock_time.return_value = start_time + 0.1
            mock_span_time.return_value = start_time + 0.1
            self.handler.on_llm_new_token("Hello", run_id=run_id)

            self.ttft_histogram.record.assert_called_once()
            args = self.ttft_histogram.record.call_args
            ttft_value = args[0][0]
            assert abs(ttft_value - 0.1) < 0.001, f"TTFT should be approximately 0.1 seconds, got {ttft_value}"

            attributes = args[1]["attributes"]
            assert attributes[SpanAttributes.LLM_SYSTEM] == "Langchain"

    def test_ttft_metric_not_recorded_on_subsequent_tokens(self):
        """Test that TTFT metric is only recorded once."""
        run_id = uuid4()
        mock_span = Mock(spec=Span)
        mock_span.attributes = {SpanAttributes.LLM_SYSTEM: "Langchain"}

        span_holder = SpanHolder(
            span=mock_span,
            token=None,
            context=None,
            children=[],
            workflow_name="test",
            entity_name="test",
            entity_path="test",
            start_time=time.time()
        )
        self.handler.spans[run_id] = span_holder

        # First token
        self.handler.on_llm_new_token("Hello", run_id=run_id)
        # Second token
        self.handler.on_llm_new_token(" world", run_id=run_id)

        # TTFT should only be recorded once
        assert self.ttft_histogram.record.call_count == 1

    def test_generation_choices_metric_recorded(self):
        """Test that generation choices metric is recorded."""
        run_id = uuid4()
        mock_span = Mock(spec=Span)
        mock_span.attributes = {SpanAttributes.LLM_SYSTEM: "Langchain"}

        span_holder = SpanHolder(
            span=mock_span,
            token=None,
            context=None,
            children=[],
            workflow_name="test",
            entity_name="test",
            entity_path="test",
            start_time=time.time()
        )
        self.handler.spans[run_id] = span_holder

        # Mock LLMResult with multiple generations
        generation1 = Generation(text="Response 1")
        generation2 = Generation(text="Response 2")
        llm_result = LLMResult(
            generations=[[generation1, generation2]],
            llm_output={"model_name": "test-model"}
        )

        self.handler.on_llm_end(llm_result, run_id=run_id)

        # Verify choices counter was called
        self.choices_counter.add.assert_called_once_with(
            2,  # Two choices
            attributes={
                SpanAttributes.LLM_SYSTEM: "Langchain",
                SpanAttributes.LLM_RESPONSE_MODEL: "test-model",
            }
        )

    def test_streaming_time_to_generate_metric(self):
        """Test that streaming time to generate metric is recorded."""
        run_id = uuid4()
        mock_span = Mock(spec=Span)
        mock_span.attributes = {SpanAttributes.LLM_SYSTEM: "Langchain"}

        with patch('opentelemetry.instrumentation.langchain.callback_handler.time.time') as mock_time, \
             patch('opentelemetry.instrumentation.langchain.span_utils.time.time') as mock_span_time:

            start_time = 1000.0
            mock_time.return_value = start_time
            mock_span_time.return_value = start_time

            span_holder = SpanHolder(
                span=mock_span,
                token=None,
                context=None,
                children=[],
                workflow_name="test",
                entity_name="test",
                entity_path="test",
                start_time=start_time
            )
            self.handler.spans[run_id] = span_holder

            first_token_time = start_time + 0.05
            mock_time.return_value = first_token_time
            mock_span_time.return_value = first_token_time
            self.handler.on_llm_new_token("Hello", run_id=run_id)

            completion_time = first_token_time + 0.05
            mock_time.return_value = completion_time
            mock_span_time.return_value = completion_time
            llm_result = LLMResult(
                generations=[[Generation(text="Hello world")]],
                llm_output={"model_name": "test-model"}
            )

            self.handler.on_llm_end(llm_result, run_id=run_id)

            self.streaming_time_histogram.record.assert_called_once()
            args = self.streaming_time_histogram.record.call_args
            streaming_time = args[0][0]
            assert abs(streaming_time - 0.05) < 0.001, (
                f"Streaming time should be approximately 0.05 seconds, "
                f"got {streaming_time}"
            )

    def test_exception_metric_recorded_on_error(self):
        """Test that exception metric is recorded on LLM errors."""
        run_id = uuid4()
        mock_span = Mock(spec=Span)
        mock_span.attributes = {SpanAttributes.LLM_SYSTEM: "Langchain"}

        span_holder = SpanHolder(
            span=mock_span,
            token=None,
            context=None,
            children=[],
            workflow_name="test",
            entity_name="test",
            entity_path="test",
            start_time=time.time()
        )
        self.handler.spans[run_id] = span_holder

        # Simulate error
        error = ValueError("API Error")
        self.handler.on_llm_error(error, run_id=run_id)

        # Verify exception counter was called
        self.exception_counter.add.assert_called_once_with(
            1,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Langchain",
                SpanAttributes.LLM_RESPONSE_MODEL: "unknown",
                "error.type": "ValueError",
            }
        )

    def test_no_ttft_when_no_first_token_time(self):
        """Test streaming time metric is not recorded without first token."""
        run_id = uuid4()
        mock_span = Mock(spec=Span)
        mock_span.attributes = {SpanAttributes.LLM_SYSTEM: "Langchain"}

        span_holder = SpanHolder(
            span=mock_span,
            token=None,
            context=None,
            children=[],
            workflow_name="test",
            entity_name="test",
            entity_path="test",
            start_time=time.time()
        )
        # No first_token_time set
        self.handler.spans[run_id] = span_holder

        llm_result = LLMResult(
            generations=[[Generation(text="Response")]],
            llm_output={"model_name": "test-model"}
        )

        self.handler.on_llm_end(llm_result, run_id=run_id)

        # Streaming time metric should not be recorded
        self.streaming_time_histogram.record.assert_not_called()
